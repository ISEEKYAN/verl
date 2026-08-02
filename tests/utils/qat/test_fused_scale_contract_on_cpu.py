import torch

from verl.utils.qat.fused_scale_contract import fuse_nvfp4_global_scales


def test_fused_nvfp4_scale_policy_is_explicitly_inverse_across_representations():
    # The two stored forms encode the same policy: a smaller reciprocal gparam
    # is the same choice as a larger divisor.
    reciprocal = fuse_nvfp4_global_scales(
        [torch.tensor([0.25]), torch.tensor([0.125])], representation="reciprocal_gparam"
    )
    divisor = fuse_nvfp4_global_scales([torch.tensor([4.0]), torch.tensor([8.0])], representation="divisor")

    torch.testing.assert_close(reciprocal, torch.tensor(0.125))
    torch.testing.assert_close(divisor, torch.tensor(8.0))
    torch.testing.assert_close(reciprocal * divisor, torch.tensor(1.0))


def test_fused_nvfp4_scale_policy_rejects_unknown_representation():
    try:
        fuse_nvfp4_global_scales([torch.tensor([1.0])], representation="ambiguous")
    except ValueError as error:
        assert "representation" in str(error)
    else:
        raise AssertionError("ambiguous NVFP4 scale representation must fail loudly")
