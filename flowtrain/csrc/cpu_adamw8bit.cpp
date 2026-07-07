#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr float kInt8Max = 127.0F;

void check_cpu_contiguous(const torch::Tensor& tensor, const char* name) {
  if (!tensor.device().is_cpu()) {
    throw std::runtime_error(std::string(name) + " must be a CPU tensor");
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(std::string(name) + " must be contiguous");
  }
}

int8_t quantize_signed(float value, float scale) {
  const float scaled = std::nearbyint(value / scale);
  const float clamped = std::max(-kInt8Max, std::min(kInt8Max, scaled));
  return static_cast<int8_t>(clamped);
}

int8_t quantize_nonnegative(float value, float scale) {
  const float nonnegative = std::max(value, 0.0F);
  float rounded = std::nearbyint(nonnegative / scale);
  rounded = std::max(0.0F, std::min(kInt8Max, rounded));
  if (nonnegative > 0.0F && rounded < 1.0F) {
    rounded = 1.0F;
  }
  return static_cast<int8_t>(rounded);
}

template <typename param_t, typename grad_t>
void adamw_step_impl(
    torch::Tensor master,
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step) {
  const int64_t numel = master.numel();

  float* master_data = master.data_ptr<float>();
  param_t* param_data = param.data_ptr<param_t>();
  const grad_t* grad_data = grad.data_ptr<grad_t>();
  float* m_data = exp_avg.data_ptr<float>();
  float* v_data = exp_avg_sq.data_ptr<float>();

  const float lr_f = static_cast<float>(lr);
  const float beta1_f = static_cast<float>(beta1);
  const float beta2_f = static_cast<float>(beta2);
  const float eps_f = static_cast<float>(eps);
  const float weight_decay_f = static_cast<float>(weight_decay);
  const float one_minus_beta1 = 1.0F - beta1_f;
  const float one_minus_beta2 = 1.0F - beta2_f;
  const float decay = 1.0F - lr_f * weight_decay_f;
  const float bias_correction1 = 1.0F - static_cast<float>(std::pow(beta1, static_cast<double>(step)));
  const float bias_correction2 = 1.0F - static_cast<float>(std::pow(beta2, static_cast<double>(step)));
  const float step_size = lr_f * std::sqrt(bias_correction2) / bias_correction1;

#pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < numel; ++idx) {
    const float g = static_cast<float>(grad_data[idx]);
    float w = master_data[idx];
    float m = m_data[idx];
    float v = v_data[idx];

    if (weight_decay_f != 0.0F) {
      w *= decay;
    }
    m = m * beta1_f + g * one_minus_beta1;
    v = v * beta2_f + g * g * one_minus_beta2;
    w += -step_size * m / (std::sqrt(v) + eps_f);

    master_data[idx] = w;
    m_data[idx] = m;
    v_data[idx] = v;
    param_data[idx] = static_cast<param_t>(w);
  }
}

template <typename param_t, typename grad_t>
void adamw8bit_step_impl(
    torch::Tensor master,
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor q_exp_avg,
    torch::Tensor exp_avg_scale,
    torch::Tensor q_exp_avg_sq,
    torch::Tensor exp_avg_sq_scale,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step,
    int64_t block_size) {
  const int64_t numel = master.numel();
  const int64_t n_blocks = (numel + block_size - 1) / block_size;

  float* master_data = master.data_ptr<float>();
  param_t* param_data = param.data_ptr<param_t>();
  const grad_t* grad_data = grad.data_ptr<grad_t>();
  int8_t* q_m_data = q_exp_avg.data_ptr<int8_t>();
  float* m_scale_data = exp_avg_scale.data_ptr<float>();
  int8_t* q_v_data = q_exp_avg_sq.data_ptr<int8_t>();
  float* v_scale_data = exp_avg_sq_scale.data_ptr<float>();

  const float lr_f = static_cast<float>(lr);
  const float beta1_f = static_cast<float>(beta1);
  const float beta2_f = static_cast<float>(beta2);
  const float eps_f = static_cast<float>(eps);
  const float weight_decay_f = static_cast<float>(weight_decay);
  const float one_minus_beta1 = 1.0F - beta1_f;
  const float one_minus_beta2 = 1.0F - beta2_f;
  const float decay = 1.0F - lr_f * weight_decay_f;
  const float bias_correction1 = 1.0F - static_cast<float>(std::pow(beta1, static_cast<double>(step)));
  const float bias_correction2 = 1.0F - static_cast<float>(std::pow(beta2, static_cast<double>(step)));
  const float step_size = lr_f * std::sqrt(bias_correction2) / bias_correction1;

#pragma omp parallel
  {
    std::vector<float> m_buffer(static_cast<size_t>(block_size));
    std::vector<float> v_buffer(static_cast<size_t>(block_size));

#pragma omp for schedule(static)
    for (int64_t block = 0; block < n_blocks; ++block) {
      const int64_t start = block * block_size;
      const int64_t end = std::min(start + block_size, numel);
      const int64_t valid = end - start;
      const float old_m_scale = m_scale_data[block];
      const float old_v_scale = v_scale_data[block];

      float m_absmax = 0.0F;
      float v_max = 0.0F;
      for (int64_t i = 0; i < valid; ++i) {
        const int64_t idx = start + i;
        const int64_t q_idx = block * block_size + i;
        const float g = static_cast<float>(grad_data[idx]);
        float m = static_cast<float>(q_m_data[q_idx]) * old_m_scale;
        float v = static_cast<float>(q_v_data[q_idx]) * old_v_scale;
        float w = master_data[idx];

        if (weight_decay_f != 0.0F) {
          w *= decay;
        }
        m = m * beta1_f + g * one_minus_beta1;
        v = std::max(0.0F, v * beta2_f + g * g * one_minus_beta2);
        w += -step_size * m / (std::sqrt(v) + eps_f);

        master_data[idx] = w;
        param_data[idx] = static_cast<param_t>(w);
        m_buffer[static_cast<size_t>(i)] = m;
        v_buffer[static_cast<size_t>(i)] = v;
        m_absmax = std::max(m_absmax, std::abs(m));
        v_max = std::max(v_max, v);
      }

      const float new_m_scale = m_absmax > 0.0F ? m_absmax / kInt8Max : 1.0F;
      const float new_v_scale = v_max > 0.0F ? v_max / kInt8Max : 1.0F;
      m_scale_data[block] = new_m_scale;
      v_scale_data[block] = new_v_scale;

      for (int64_t i = 0; i < valid; ++i) {
        const int64_t q_idx = block * block_size + i;
        q_m_data[q_idx] = quantize_signed(m_buffer[static_cast<size_t>(i)], new_m_scale);
        q_v_data[q_idx] = quantize_nonnegative(v_buffer[static_cast<size_t>(i)], new_v_scale);
      }
      for (int64_t i = valid; i < block_size; ++i) {
        const int64_t q_idx = block * block_size + i;
        q_m_data[q_idx] = 0;
        q_v_data[q_idx] = 0;
      }
    }
  }
}

}  // namespace

void adamw_step(
    torch::Tensor master,
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step) {
  check_cpu_contiguous(master, "master");
  check_cpu_contiguous(param, "param");
  check_cpu_contiguous(grad, "grad");
  check_cpu_contiguous(exp_avg, "exp_avg");
  check_cpu_contiguous(exp_avg_sq, "exp_avg_sq");
  if (master.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("master must be float32");
  }
  if (exp_avg.scalar_type() != torch::kFloat32 || exp_avg_sq.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("AdamW moments must be float32");
  }
  if (param.numel() != master.numel() || grad.numel() != master.numel() ||
      exp_avg.numel() != master.numel() || exp_avg_sq.numel() != master.numel()) {
    throw std::runtime_error("AdamW tensors must have matching numel");
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, param.scalar_type(), "adamw_step_param", [&] {
        using param_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf, grad.scalar_type(), "adamw_step_grad", [&] {
              using grad_t = scalar_t;
              adamw_step_impl<param_t, grad_t>(
                  master,
                  param,
                  grad,
                  exp_avg,
                  exp_avg_sq,
                  lr,
                  beta1,
                  beta2,
                  eps,
                  weight_decay,
                  step);
            });
      });
}

void adamw8bit_step(
    torch::Tensor master,
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor q_exp_avg,
    torch::Tensor exp_avg_scale,
    torch::Tensor q_exp_avg_sq,
    torch::Tensor exp_avg_sq_scale,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step,
    int64_t block_size) {
  if (block_size < 1) {
    throw std::runtime_error("block_size must be >= 1");
  }
  check_cpu_contiguous(master, "master");
  check_cpu_contiguous(param, "param");
  check_cpu_contiguous(grad, "grad");
  check_cpu_contiguous(q_exp_avg, "q_exp_avg");
  check_cpu_contiguous(exp_avg_scale, "exp_avg_scale");
  check_cpu_contiguous(q_exp_avg_sq, "q_exp_avg_sq");
  check_cpu_contiguous(exp_avg_sq_scale, "exp_avg_sq_scale");
  if (master.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("master must be float32");
  }
  if (q_exp_avg.scalar_type() != torch::kInt8 || q_exp_avg_sq.scalar_type() != torch::kInt8) {
    throw std::runtime_error("quantized moments must be int8");
  }
  if (exp_avg_scale.scalar_type() != torch::kFloat32 || exp_avg_sq_scale.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("moment scales must be float32");
  }
  if (param.numel() != master.numel() || grad.numel() != master.numel()) {
    throw std::runtime_error("master, param, and grad must have matching numel");
  }
  const int64_t n_blocks = (master.numel() + block_size - 1) / block_size;
  if (q_exp_avg.numel() < n_blocks * block_size || q_exp_avg_sq.numel() < n_blocks * block_size) {
    throw std::runtime_error("quantized moment tensors are too small");
  }
  if (exp_avg_scale.numel() < n_blocks || exp_avg_sq_scale.numel() < n_blocks) {
    throw std::runtime_error("moment scale tensors are too small");
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, param.scalar_type(), "adamw8bit_step_param", [&] {
        using param_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf, grad.scalar_type(), "adamw8bit_step_grad", [&] {
              using grad_t = scalar_t;
              adamw8bit_step_impl<param_t, grad_t>(
                  master,
                  param,
                  grad,
                  q_exp_avg,
                  exp_avg_scale,
                  q_exp_avg_sq,
                  exp_avg_sq_scale,
                  lr,
                  beta1,
                  beta2,
                  eps,
                  weight_decay,
                  step,
                  block_size);
            });
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adamw_step", &adamw_step, "Fused fp32-state CPU AdamW step");
  m.def("adamw8bit_step", &adamw8bit_step, "Fused block-wise int8 CPU AdamW step");
}
