// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ov_utils.h"
#include "ov_factory.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

class OvDataTransfer : public OrtDataTransferImpl {
 public:
  explicit OvDataTransfer(Ort::Logger logger);
  ~OvDataTransfer() = default;

  OVEP_DISABLE_COPY_AND_MOVE(OvDataTransfer)

  Ort::Logger GetLogger() const { return logger_; }

  void Release() noexcept { delete this; }

  bool CanCopy(const OrtMemoryDevice* src_device,
               const OrtMemoryDevice* dst_device) const noexcept;

  OrtStatus* CopyTensors(const OrtValue** src_tensors,
                         OrtValue** dst_tensors,
                         OrtSyncStream** streams,
                         size_t count);

 private:
  const OrtApi& ort_api_{Ort::GetApi()};
  const OrtEpApi& ep_api_{Ort::GetEpApi()};
  Ort::Logger logger_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
