from math import ceil
from math import floor

params = 2
p = 4

if params == 1:
    k = 4
    l = 4
    r = 4.25
    w_rounds = 7
    pk_rounds = 10
    pr = (1 / r)
elif params == 2:
    k = 6
    l = 5
    r = 5.1 
    w_rounds = 7
    pk_rounds = 15
    pr = (1 / r)
elif params == 3:
    k = 8
    l = 7
    r = 3.85
    w_rounds = 9
    pk_rounds = 20
    pr = (1 / r)


def keccak_instance_for_y(rounds):
    loop_size = (k * l - 1) % p
    count = 0
    for i in range(1, rounds + 1):
        count = 0
        while loop_size < l:
            count += 1
            loop_size += p
        loop_size -= l
        loop_size += 3
    return count


def expected_instance_for_y(max_rounds=10**4):
    """
    tol        : truncation tolerance
    f_max   : callable, upper bound of f(i), f(i) <= f_max(i)
    max_rounds : safety cap

    returns E[C]
    """
    q = 1.0 - pr
    expected = 0.0
    weight = 1.0  # q^(i-1)

    for i in range(1, max_rounds + 1):
        expected += keccak_instance_for_y(i) * weight
        weight *= q

    return expected


ori_sign_invoked_rounds = (
    (5 * ceil((k * l) / 4) + 1 + 1) * p
    + (r * ceil(l / 4) * 5) * p
    + r * (1 + w_rounds) * p
)
sign_valid_rounds = 5 * (k * l) + 1 + 1 + r * (l * 5 + w_rounds + 1)
print("sign_valid_rounds: ", sign_valid_rounds)
print("ori_sign_invoked_rounds: ", ori_sign_invoked_rounds)
ori_sign_ur = sign_valid_rounds / ori_sign_invoked_rounds
print("ori_sign_ur: ", ori_sign_ur)

expected_instance = expected_instance_for_y()
print("expected_instance_for_y: ", expected_instance)
opti_sign_invoked_rounds = (
    (5 + 1 + 5 * ceil((k * l - 3) / 4)) * p  # ssa1 + \rho'' + A + hsh
    + expected_instance * 5 * p
    + r * (w_rounds + 1) * p
)
print("opti_sign_invoked_rounds: ", opti_sign_invoked_rounds)
opti_sign_ur = sign_valid_rounds/ opti_sign_invoked_rounds
print("opti_sign_ur: ", opti_sign_ur)

print("vs",opti_sign_invoked_rounds-ori_sign_invoked_rounds," / ", (opti_sign_ur-ori_sign_ur)*100 )



#Verify


ori_verify_invoked_rounds = (5 * ceil((k * l) / 4) + pk_rounds + 1 + 1 + w_rounds) * p
verify_valid_rounds = 5 * (k * l) + pk_rounds + 1 + 1 + w_rounds

print("verify_valid_rounds: ", verify_valid_rounds)
print("ori_verify_invoked_rounds: ", ori_verify_invoked_rounds)
ori_verify_ur = verify_valid_rounds / ori_verify_invoked_rounds
print("ori_verify_ur: ", ori_verify_ur)


opti_verify_invoked_rounds = (
    pk_rounds + 5 * ceil(((k * l) - (pk_rounds / 5) * 3) / 4) + 1 * (params//3) + w_rounds
) * p

print("opti_verify_invoked_rounds: ", opti_verify_invoked_rounds)
opti_verify_ur = verify_valid_rounds / opti_verify_invoked_rounds
print("opti_verify_ur: ", opti_verify_ur)

print("vs",opti_verify_invoked_rounds-ori_verify_invoked_rounds," / ", (opti_verify_ur-ori_verify_ur)*100 )


