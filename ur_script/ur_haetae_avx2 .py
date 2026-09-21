from math import ceil
from math import floor

params = 1
p = 4
rs = 10
prs = 0.1

if params == 1:
    k = 2
    l = 4
    r = 6
    w_rounds = 5
    pk_rounds = 8
    pkA = 2
    pr = (1 / r)
    d = 1
elif params == 2:
    k = 3
    l = 6
    r = 5
    w_rounds = 7
    pk_rounds = 11
    pkA = 3
    pr = (1 / r)
    d = 1
elif params == 3:
    k = 4
    l = 7
    r = 6
    w_rounds = 9
    pk_rounds = 16
    pkA = 4
    pr = (1 / r)
    d = 0
    


m = l - 1


def keccak_instance_for_y(rounds):
    loop_size = 0
    count = 0
    for i in range(1, rounds + 1):
        count = 0
        while loop_size < l + k:
            count += 1
            loop_size += p
        loop_size -= (l + k)
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


def keccak_instance_for_s(rounds):
    if params == 2 :
        loop_size = 8
    else:
        loop_size = 0
    count = 0
    for i in range(1, rounds + 1):
        count = 0
        while loop_size < k + m:
            count += 1
            loop_size += p
        loop_size -= (k + m)
    return count


def expected_instance_for_s(max_rounds=10**4):
    q = 1.0 - prs
    expected = 0.0
    weight = 1.0  # q^(i-1)

    for i in range(1, max_rounds + 1):
        expected += keccak_instance_for_s(i) * weight
        weight *= q

    return expected

#KeyGen
keygen_valid_rounds = 1 + 4 * k * m + 4 * d * k + (m + k) * rs
ori_keygen_invoked_rounds = (
    1 + 4 * ceil((k * m)/p) + 4 * d + ceil((m + k)/p) * rs
) * p
expected_instance_s = expected_instance_for_s()
print("expected_instance_for_s: ", expected_instance_s)
opti_keygen_invoked_rounds = (
    1 + 4 * ceil((k * m + d * k)/p) 
) * p + expected_instance_s * p
print("keygen_valid_rounds", keygen_valid_rounds)
print("ori_keygen_invoked_rounds", ori_keygen_invoked_rounds)
ori_keygen_ur = keygen_valid_rounds/ori_keygen_invoked_rounds
print("ori_keygen_ur", ori_keygen_ur)
opti_keygen_ur = keygen_valid_rounds/opti_keygen_invoked_rounds
print("opti_keygen_invoked_rounds", opti_keygen_invoked_rounds)
print("opti_keygen_ur", opti_keygen_ur)
print("vs",opti_keygen_invoked_rounds-ori_keygen_invoked_rounds," / ", (opti_keygen_ur-ori_keygen_ur)*100 )

#Sign

sign_valid_rounds = pk_rounds + 4 * (k * l) + 1 + r * ((l + k) * 35 + w_rounds + 1)
ori_sign_invoked_rounds = (
    pk_rounds * p
    + (4 * (ceil((k * l) / p)) + 1) * p
    + r * (ceil((l + k) / p) * 35 + 1 + w_rounds) * p
)
print("sign_valid_rounds: ", sign_valid_rounds)
print("ori_sign_invoked_rounds: ", ori_sign_invoked_rounds)
ori_sign_ur = sign_valid_rounds / ori_sign_invoked_rounds
print("ori_sign_ur: ", ori_sign_ur)

expected_instance = expected_instance_for_y()
print("expected_instance_for_y: ", expected_instance)
opti_sign_invoked_rounds = (
    pk_rounds * p
    + (4 * (ceil((k * l - pkA * 3) / p)) + 1) * p  # ssa1 + \rho'' + A + hsh
    + expected_instance * 35 * p
    + r * (w_rounds + 1) * p
)
print("opti_sign_invoked_rounds: ", opti_sign_invoked_rounds)
opti_sign_ur = sign_valid_rounds/ opti_sign_invoked_rounds
print("opti_sign_ur: ", opti_sign_ur)

print("vs",opti_sign_invoked_rounds-ori_sign_invoked_rounds," / ", (opti_sign_ur-ori_sign_ur)*100 )

#Verify

verify_valid_rounds = 4 * (k * l) + 4 * k * d + pk_rounds + 1 + w_rounds
ori_verify_invoked_rounds = (4 * ceil((k * l) / p) + d + pk_rounds + 1 + w_rounds) * p

print("verify_valid_rounds: ", verify_valid_rounds)
print("ori_verify_invoked_rounds: ", ori_verify_invoked_rounds)
ori_verify_ur = verify_valid_rounds / ori_verify_invoked_rounds
print("ori_verify_ur: ", ori_verify_ur)


opti_verify_invoked_rounds = (
    pk_rounds + (4 * (ceil((k * l - pkA * 3) / p))) + 1 + w_rounds
) * p

print("opti_verify_invoked_rounds: ", opti_verify_invoked_rounds)
opti_verify_ur = verify_valid_rounds / opti_verify_invoked_rounds
print("opti_verify_ur: ", opti_verify_ur)

print("vs",opti_verify_invoked_rounds-ori_verify_invoked_rounds," / ", (opti_verify_ur-ori_verify_ur)*100 )