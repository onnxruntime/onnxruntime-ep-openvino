// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ov_data_transfer.h"

#include <cstring>

#include "ort_api_version.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

OvDataTransfer::OvDataTransfer(Ort::Logger logger) : OrtDataTransferImpl{}, logger_(logger) {
  ort_version_supported = ort_runtime_api_version;
  OrtDataTransferImpl::Release = OVEP_API_ENTRY(OvDataTransfer, Release);
  OrtDataTransferImpl::CanCopy = OVEP_API_ENTRY(OvDataTransfer, CanCopy);
  OrtDataTransferImpl::CopyTensors = OVEP_API_ENTRY(OvDataTransfer, CopyTensors);
}

bool OvDataTransfer::CanCopy(const OrtMemoryDevice* src_device,
                             const OrtMemoryDevice* dst_device) const noexcept {
  auto src_type = ep_api_.MemoryDevice_GetDeviceType(src_device);
  auto dst_type = ep_api_.MemoryDevice_GetDeviceType(dst_device);

  bool src_is_cpu = (src_type == OrtMemoryInfoDeviceType_CPU);
  bool dst_is_cpu = (dst_type == OrtMemoryInfoDeviceType_CPU);
  bool src_is_intel_device = !src_is_cpu &&
                             ep_api_.MemoryDevice_GetVendorId(src_device) == OpenVINOEpPluginFactory::vendor_id_;
  bool dst_is_intel_device = !dst_is_cpu &&
                             ep_api_.MemoryDevice_GetVendorId(dst_device) == OpenVINOEpPluginFactory::vendor_id_;

  // Support: CPU↔CPU, CPU↔Intel device, Intel device↔Intel device
  return (src_is_cpu && dst_is_cpu) ||
         (src_is_cpu && dst_is_intel_device) ||
         (src_is_intel_device && dst_is_cpu) ||
         (src_is_intel_device && dst_is_intel_device);
}

OrtStatus* OvDataTransfer::CopyTensors(const OrtValue** src_tensors,
                                       OrtValue** dst_tensors,
                                       OrtSyncStream** /*streams*/,
                                       size_t count) {
  for (size_t i = 0; i < count; ++i) {
    Ort::ConstValue src{src_tensors[i]};
    Ort::UnownedValue dst{dst_tensors[i]};

    if (src.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      const size_t count_elements = src.GetTensorTypeAndShapeInfo().GetElementCount();
      if (count_elements == 0) continue;

      if (dst.GetTensorMutableRawData() != src.GetTensorRawData()) {
        OVEP_ENFORCE(dst.GetTensorTypeAndShapeInfo().GetElementCount() >= count_elements,
                     "Destination tensor buffer is too small", count_elements, " elements required but only ",
                     dst.GetTensorTypeAndShapeInfo().GetElementCount(), " elements available");
        for (size_t elem_idx = 0; elem_idx < count_elements; ++elem_idx) {
          dst.FillStringTensorElement(src.GetStringTensorElement(elem_idx).c_str(), elem_idx);
        }
      }
      continue;
    }

    const size_t bytes = src.GetTensorSizeInBytes();
    if (bytes == 0) continue;

    const void* src_data = src.GetTensorRawData();
    void* dst_data = dst.GetTensorMutableRawData();

    if (dst_data != src_data) {
      OVEP_ENFORCE(dst.GetTensorSizeInBytes() >= bytes, "Destination tensor buffer is too small", bytes, " bytes required but only ", dst.GetTensorSizeInBytes(), " bytes available");
      std::memcpy(dst_data, src_data, bytes);
    }
  }

  return nullptr;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
