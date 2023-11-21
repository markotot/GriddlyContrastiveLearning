import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))
@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

N_ITERS = 10

from torchvision.models import densenet121
def init_model():
    return densenet121().to(torch.float32).cuda()

def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

if __name__ == "__main__":
    model = init_model()

    # Reset since we are using a different mode.
    import torch._dynamo

    torch._dynamo.reset()

    model_opt = torch.compile(model, mode="reduce-overhead")

    inp = generate_data(16)[0]
    with torch.no_grad():
        print("eager:", timed(lambda: model(inp))[1])
        print("compile:", timed(lambda: model_opt(inp))[1])

    eager_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        with torch.no_grad():
            _, eager_time = timed(lambda: model(inp))
        eager_times.append(eager_time)
        print(f"eager eval time {i}: {eager_time}")

    print("~" * 10)

    compile_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        with torch.no_grad():
            _, compile_time = timed(lambda: model_opt(inp))
        compile_times.append(compile_time)
        print(f"compile eval time {i}: {compile_time}")
    print("~" * 10)

    import numpy as np

    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    assert (speedup > 1)
    print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print("~" * 10)