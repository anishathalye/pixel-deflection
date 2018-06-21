# Deflecting Adversarial Attacks with Pixel Deflection

The code in this repository demonstrates that [Deflecting Adversarial
Attacks with Pixel Deflection](https://arxiv.org/abs/1801.08926) (Prakash et al. 2018) is ineffective
in the white-box threat model.

With an L-infinity perturbation of 4/255, we generate targeted adversarial
examples with 97% success rate, and can reduce classifier accuracy to
0%.

See [our note](https://arxiv.org/abs/1804.03286) for more context and details.

## Citation

```
@unpublished{cvpr2018breaks,
  author = {Anish Athalye and Nicholas Carlini},
  title = {On the Robustness of the CVPR 2018 White-Box Adversarial Example Defenses},
  year = {2018},
  url = {https://arxiv.org/abs/1804.03286},
}
```

## [robustml] evaluation

Run with:

```bash
python robustml_attack.py --imagenet-path <path>
````

[robustml]: https://github.com/robust-ml/robustml

### Credits

Thanks to [Nicholas Carlini](https://github.com/carlini/pixel-deflection) for
implementing the break and [Dimitris Tsipras](https://github.com/dtsip) for
writing the robustml model wrapper.
