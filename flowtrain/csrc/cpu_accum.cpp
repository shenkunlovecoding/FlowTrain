#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

void accumulate_grad_slab(
    std::vector<torch::Tensor> grads,
    torch::Tensor slab,
    std::vector<int64_t> numels) {
  if (grads.size() != numels.size()) {
    throw std::runtime_error("grads and numels must have the same length");
  }
  if (!slab.device().is_cpu()) {
    throw std::runtime_error("slab must be a CPU tensor");
  }

  int64_t offset = 0;
  for (size_t i = 0; i < grads.size(); ++i) {
    torch::Tensor grad = grads[i];
    if (!grad.device().is_cpu()) {
      throw std::runtime_error("grad tensors must be CPU tensors");
    }
    const int64_t numel = numels[i];
    torch::Tensor src = slab.narrow(0, offset, numel).view(grad.sizes());
    if (src.scalar_type() != grad.scalar_type()) {
      src = src.to(grad.scalar_type());
    }
    grad.add_(src);
    offset += numel;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("accumulate_grad_slab", &accumulate_grad_slab, "Accumulate a flat grad slab into CPU grad tensors");
}
