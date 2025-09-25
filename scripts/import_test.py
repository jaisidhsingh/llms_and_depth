from src.utils import *
from src.tracer import *
from src.cli import Args

print("Imports work.")
if __name__ == "__main__":
    model = get_model("llama-1b", "cuda", on_colab=False)
    print("Model loads fine.")
