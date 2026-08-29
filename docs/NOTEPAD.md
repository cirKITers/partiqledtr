# Notepad

Small intermediate bugs/features land here (see README note).

## qml-essentials

**Open (0.2.3): `Model.apply`'s docstring names one axis fewer than it returns.**
Dropping the squeeze was deliberate; the axis list beside it was not updated. The
docstring promises `(B_I, B_P, B_R, O)` -- four axes -- and the method returns five.

Measured on `make_qfm(...)`: inputs `(7, 4)` with params `(1, 3, 6)` give
`(7, 1, 1, 1, 4)`, params `(2, 3, 6)` give `(7, 2, 1, 1, 4)`. So axis 0 is `B_I`,
axis 1 is `B_P`, axis -1 is `O`, and two singleton axes sit where the docstring
names one `B_R`. Naming both would fix it. The rest of that paragraph still holds:
`__call__` does squeeze, and returns `(7, 4)` for the same call.

Callers that reshape or squeeze are unaffected -- `g_purity_model`'s
`execution_type="state"` path here reshapes to `(-1, 2**n)` and did not notice.
Ones that assert the documented shape are: `models/qfm.py` `_edges` did, and now
pins the five-axis rank instead.

## phasespace-jax


## Fluksio

