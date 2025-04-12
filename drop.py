import torch
# import triton.tools.experimental_descriptor

import triton
import triton.language as tl
import tabulate


# Exercises
#     Extend the kernel to operate over a matrix and use a vector of seeds - one per row.
#     Add support for striding.
#     (challenge) Implement a kernel for sparse Johnson-Lindenstrauss transform which generates the projection matrix on the fly each time using a seed.


@triton.jit
def _seeded_dropout(
    x_ptr,
    seed_ptr,
    output_ptr,
    p,
    stride: tl.constexpr,  # Number of elements in a row
    BLOCK_M: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    seed = tl.load(seed_ptr + pid)

    read_block_start = x_ptr + stride * pid
    write_block_start = output_ptr + stride * pid

    for i in range(tl.cdiv(stride, BLOCK_M)):
        offset = i * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offset < stride

        x = tl.load(read_block_start + offset, mask=mask)

        random = tl.rand(seed, offset)
        keep = random > p

        output = tl.where(keep, x / (1 - p), 0)
        tl.store(write_block_start + offset, output, mask=mask)


def seeded_dropout(x, seeds, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()

    def grid(_):
        return (x.shape[1],)

    _seeded_dropout[grid](x, seeds, output, p, stride=x.shape[0], BLOCK_M=32)
    return output


if __name__ == "__main__":
    a, b = 16, 8

    x = torch.randint(9, size=(a, b), device=torch.device("cuda")) + 1
    seeds = torch.randint(0, 1000, size=(b,), device=torch.device("cuda"))

    # Compare this to the baseline - dropout mask is never instantiated!
    output = seeded_dropout(x, seeds, p=0.5)
    output_2 = seeded_dropout(x, seeds, p=0.5)

    print(
        tabulate.tabulate(
            [
                ["input"] + x.tolist(),
                ["output (seed = 123)"] + output.tolist(),
            ]
        )
    )
