import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 3. Coding attention mechanism""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3.3.1 A simple self-attention mechanism without trainable weight""")
    return


@app.cell
def _():
    import torch

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your
            [0.55, 0.87, 0.66],  # journey
            [0.57, 0.85, 0.64],  # starts
            [0.22, 0.58, 0.33],  # with
            [0.77, 0.25, 0.10],  # one
            [0.05, 0.80, 0.55],  # step
        ]
    )
    return inputs, torch


@app.cell
def _(inputs, torch):
    query_1 = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for _i, _x_i in enumerate(inputs):
        attn_scores_2[_i] = torch.dot(_x_i, query_1)
    print(attn_scores_2)
    return attn_scores_2, query_1


@app.cell
def _(attn_scores_2):
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())
    return


@app.cell
def _(attn_scores_2, torch):
    def softmax_naive(x):
        return torch.exp(x) / torch.exp(x).sum(dim=0)


    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())
    return


@app.cell
def _(attn_scores_2, torch):
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())
    return (attn_weights_2,)


@app.cell
def _(attn_weights_2, inputs, query_1, torch):
    context_vec_2 = torch.zeros(query_1.shape)
    for _i, _x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[_i] * _x_i
    print(context_vec_2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
