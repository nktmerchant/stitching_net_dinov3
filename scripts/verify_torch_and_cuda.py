import torch
import subprocess


def main():
    print("+" + "-" * 89 + "+")
    print(f"torch version: {torch.__version__}")
    print(f"cuda version: {torch.version.cuda}")

    if torch.cuda.is_available():
        print("cuda is available")
        print(f"gpu count: {torch.cuda.device_count()}")
        print("+" + "-" * 89 + "+")

        try:
            x = torch.rand((1000, 1))
            x.to("cuda")
            subprocess.run(["nvidia-smi"])
        except:
            print("failed to allocate tensor to device")

    else:
        print("+" + "-" * 89 + "+")
        print("cuda is unavailable")


if __name__ == "__main__":
    main()
