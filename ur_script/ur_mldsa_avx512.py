from math import ceil
from math import floor

params = 1
p = 8

if params == 1:
    k = 4
    l = 4
    r = 4.25
    w_rounds = 7
    pk_rounds = 10
    pr = 1 / r
elif params == 2:
    k = 6
    l = 5
    r = 5.1
    w_rounds = 7
    pk_rounds = 15
    pr = 1 / r
elif params == 3:
    k = 8
    l = 7
    r = 3.85
    w_rounds = 9
    pk_rounds = 20
    pr = 1 / r


sign_valid_rounds = 5 * (k * l) + 1 + 1 + r * (l * 5 + w_rounds + 1)
ori_sign_invoked_rounds = (
    (5 * ceil((k * l) / p) + 1 + 1) * p
    + (r * ceil(l / p) * 5) * p
    + r * (1 + w_rounds) * p
)
print("sign_valid_rounds: ", sign_valid_rounds)
print("ori_sign_invoked_rounds: ", ori_sign_invoked_rounds)
ori_sign_ur = sign_valid_rounds / ori_sign_invoked_rounds
print("ori_sign_ur: ", ori_sign_ur)


if params == 1:
    opti_sign_invoked_rounds = (
        (5 + 5 + 5 * ceil((k * l - 14) / p)) * p 
        + r * (w_rounds + 1) * p
    )
elif params == 2:
    opti_sign_invoked_rounds = (
        ((5 + 5 + 5 * ceil((k * l - 14) / p)) * p 
        + 5 * p
        + r * (w_rounds + 1) * p)
    )
elif params == 3:
    opti_sign_invoked_rounds = (
        (5 + 1 + 5 * ceil((k * l - 7) / p)) * p 
        + r * (w_rounds + 1) * p
    )

print("opti_sign_invoked_rounds: ", opti_sign_invoked_rounds)
opti_sign_ur = sign_valid_rounds / opti_sign_invoked_rounds
print("opti_sign_ur: ", opti_sign_ur)

print(
    "vs",
    opti_sign_invoked_rounds - ori_sign_invoked_rounds,
    " / ",
    (opti_sign_ur - ori_sign_ur) * 100,
)

# Verify


ori_verify_invoked_rounds = (5 * ceil((k * l) / p) + pk_rounds + 1 + 1 + w_rounds) * p
verify_valid_rounds = 5 * (k * l) + pk_rounds + 1 + 1 + w_rounds

print("verify_valid_rounds: ", verify_valid_rounds)
print("ori_verify_invoked_rounds: ", ori_verify_invoked_rounds)
ori_verify_ur = verify_valid_rounds / ori_verify_invoked_rounds
print("ori_verify_ur: ", ori_verify_ur)


opti_verify_invoked_rounds = (
    pk_rounds + 5 * ceil(((k * l) - (pk_rounds / 5) * 7) / p) + w_rounds
) * p

print("opti_verify_invoked_rounds: ", opti_verify_invoked_rounds)
opti_verify_ur = verify_valid_rounds / opti_verify_invoked_rounds
print("opti_verify_ur: ", opti_verify_ur)

print(
    "vs",
    opti_verify_invoked_rounds - ori_verify_invoked_rounds,
    " / ",
    (opti_verify_ur - ori_verify_ur) * 100,
)
