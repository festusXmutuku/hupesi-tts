

def Vieneu(mode="v3turbo", **kwargs):
    """
    Factory function for VieNeu-TTS.

    Args:
        mode: 'v3turbo' (DEFAULT) — VieNeu-TTS v3 Turbo, 48 kHz. CPU runs torch-free
              via ONNX Runtime; GPU uses PyTorch. Works with the minimal install.
              'v3nano' — VieNeu-TTS v3 Nano, 24 kHz. Also torch-free ONNX on CPU;
              a smaller flow-matching model for machines where Turbo is too slow.
              Legacy v1/v2 modes (standard, fast, turbo, remote, xpu) have been removed from
              the desktop build; use the original ``vieneu`` package on PyPI if needed.
        **kwargs: Arguments for chosen class

    Returns:
        BaseVieneuTTS: An instance of a VieNeu-TTS implementation.
    """
    match mode:
        case "v3nano":
            from .v3nano import V3NanoVieNeuTTS
            return V3NanoVieNeuTTS(**kwargs)
        case "v3turbo":
            from .v3turbo import V3TurboVieNeuTTS
            return V3TurboVieNeuTTS(**kwargs)

        case _:
            raise ValueError(f"Unsupported mode: {mode!r}. Use 'v3turbo' or 'v3nano'.")
