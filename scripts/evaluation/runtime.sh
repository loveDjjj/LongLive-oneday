#!/usr/bin/env bash
# 内部 helper：统一评测入口的加速器、可见设备和环境选择。

evaluation_runtime_init() {
  export LLV2_DEVICE="${LLV2_DEVICE:-npu}"
  [[ "${LLV2_DEVICE}" == "ascend" ]] && LLV2_DEVICE=npu
  case "${LLV2_DEVICE}" in
    cuda)
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
      VISIBLE_DEVICES_VARIABLE=CUDA_VISIBLE_DEVICES
      if [[ -z "${GENERATION_ENV:-}" ]]; then
        local active_python
        active_python="$(command -v python || command -v python3)"
        export GENERATION_ENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-$("${active_python}" -c 'import sys; print(sys.prefix)')}}"
      fi
      ;;
    npu)
      export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${1:-0,1,2,3,4}}"
      VISIBLE_DEVICES_VARIABLE=ASCEND_RT_VISIBLE_DEVICES
      export GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
      export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh}"
      ;;
    *) echo "[error] LLV2_DEVICE must be npu or cuda" >&2; return 2 ;;
  esac
  export GENERATION_ENV
  VISIBLE_DEVICES="${!VISIBLE_DEVICES_VARIABLE}"
}

evaluation_prepare_runtime() {
  if [[ ! -x "${GENERATION_ENV}/bin/python" ]]; then
    echo "[error] missing generation Python: ${GENERATION_ENV}/bin/python" >&2
    return 1
  fi
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if [[ ! -x "${GENERATION_ENV}/bin/torchrun" ]]; then
      echo "[error] missing torchrun: ${GENERATION_ENV}/bin/torchrun" >&2
      return 1
    fi
    if [[ "${LLV2_DEVICE}" == "npu" ]]; then
      if [[ ! -f "${CANN_ENV_SCRIPT}" ]]; then
        echo "[error] missing CANN environment: ${CANN_ENV_SCRIPT}" >&2
        return 1
      fi
      set +u
      # shellcheck disable=SC1090
      source "${CANN_ENV_SCRIPT}"
      set -u
    fi
  fi
  export PATH="${GENERATION_ENV}/bin:${PATH}"
  export LD_LIBRARY_PATH="${GENERATION_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}
