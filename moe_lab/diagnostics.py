"""Stage comparisons that preserve failure evidence without aborting collectives."""
import torch


def tensor_comparison(actual, expected, *, atol=0., rtol=0.):
    report = {'actual_shape': list(actual.shape), 'expected_shape': list(expected.shape),
              'atol': atol, 'rtol': rtol, 'passed': False}
    if actual.shape != expected.shape:
        return dict(report, reason='shape')
    a, b = actual.detach().double(), expected.detach().double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    report['nonfinite_elements'] = int((~finite).sum())
    if not bool(finite.all()):
        return dict(report, reason='nonfinite', max_abs_error=None)
    delta = (a - b).abs()
    failed = delta > atol + rtol * b.abs()
    report.update(passed=not bool(failed.any()), mismatched_elements=int(failed.sum()),
                  max_abs_error=float(delta.max()) if delta.numel() else 0.,
                  relative_l2_error=float(torch.linalg.vector_norm(a-b) /
                                          torch.linalg.vector_norm(b).clamp_min(1e-30)))
    if bool(failed.any()):
        report['first_mismatch'] = torch.nonzero(failed)[0].tolist()
    return report


def first_failed_stage(stages):
    """Input order defines the execution-stage order, not error magnitude."""
    return next((name for name, comparisons in stages.items()
                 if not all(c['passed'] for c in comparisons)), None)
