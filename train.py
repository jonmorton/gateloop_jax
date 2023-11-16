import os
import time

import draccus
import equinox as eqx
import jax
import jmp
import optax
from tensorboardX import SummaryWriter

from linseq.config import Config
from linseq.data import DatasetManager

policy = jmp.Policy(
    param_dtype=jax.numpy.float32,
    compute_dtype=jax.numpy.bfloat16,
    output_dtype=jax.numpy.float32,
)


def compute_loss(model, x):
    model = policy.cast_to_compute(model)
    y = x[:, 1:]
    x = x[:, :-1]
    logits = jax.vmap(model)(x)
    pred = jax.numpy.argmax(logits, axis=-1)
    correct = pred == y
    lse = logits - jax.nn.logsumexp(logits)
    yhot = jax.nn.one_hot(y, num_classes=logits.shape[-1], dtype=jax.numpy.float32)
    return -jax.numpy.mean(lse * yhot), (pred, jax.numpy.sum(correct) / correct.size)


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)


@eqx.filter_jit
def step(model, optimizer, optim_state, batch):
    (loss, (pred, correct)), grads = compute_loss_and_grads(model, batch)
    updates, optim_state = optimizer.update(grads, optim_state, model)
    model = eqx.apply_updates(model, updates)
    return model, optim_state, loss, pred, correct


@eqx.filter_jit
def eval_step(model, batch):
    loss, (_, correct) = compute_loss(model, batch)
    return loss, correct


def train(cfg: Config):
    max_iter = cfg.optim.total_steps
    start_iter = 0

    os.makedirs(cfg.out_dir, exist_ok=True)
    draccus.dump(cfg, open(os.path.join(cfg.out_dir, "config.yaml"), "w"))

    tb = SummaryWriter(log_dir=os.path.join(cfg.out_dir, "tb"))

    data_mgr = DatasetManager(cfg.data.root)

    if cfg.model.vocab_size == -1:
        cfg.model.vocab_size = data_mgr.vocab_size

    model = cfg.model.build(key=None)
    model = policy.cast_to_param(model)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optim.grad_norm_clip),
        optax.adamw(
            cfg.optim.lr,
            b1=cfg.optim.beta1,
            b2=cfg.optim.beta2,
            weight_decay=cfg.optim.weight_decay,
            eps=cfg.optim.eps,
        ),
    )
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    tokens_per_batch = cfg.model.seq_len * cfg.optim.batch_size
    ntok = 0

    for iteration, data in zip(
        range(start_iter, max_iter),
        data_mgr.train_iter(cfg.optim.batch_size, cfg.model.seq_len + 1),
    ):
        # nt = time.perf_counter()
        # data_time = nt - t

        t = time.perf_counter()

        model, optimizer_state, loss, pred, correct = step(
            model, optimizer, optimizer_state, data
        )
        accuracy = correct.sum().item() / correct.size
        loss = loss.item()
        model_time = time.perf_counter() - t

        print(loss, accuracy, model_time)

        ntok += tokens_per_batch

        if iteration > 0 and iteration % cfg.val_period == 0:
            plen = 128
            vp = pred[0, :plen]
            vt = data[0, 1 : plen + 1]

            print("pred>", data_mgr.decode(vp.tolist()))
            print("targ>", data_mgr.decode(vt.tolist()))

        if iteration > 0 and iteration % cfg.val_period == 0:
            total_loss = 0
            total_acc = 0
            count = 0
            for i, batch in zip(
                range(cfg.val_examples // cfg.optim.batch_size),
                data_mgr.validation_iter(cfg.optim.batch_size, cfg.model.seq_len + 1),
            ):
                loss, correct = eval_step(model, batch)
                accuracy = correct.sum().item() / correct.size
                total_loss += loss.item()
                total_acc += accuracy
                count += 1

            tb.add_scalar("loss/validation", total_loss / count, iteration)
            tb.add_scalar("accuracy/validation", total_acc / count, iteration)

        # if iteration > 0 and iteration % cfg.log_hists_period == 0:
        #     for name, p in model.named_parameters():
        #         nnt.log({f"parameters/{name}": p.data.to("cpu")})


@draccus.wrap()
def main(cfg: Config):
    train(cfg)


if __name__ == "__main__":
    main()
