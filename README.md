[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/moritzhuetten/dmbounds/HEAD?labpath=tutorial.ipynb)

# DMbounds

A database for all latest indirect Dark Matter limits, and to compare against direct and collider limits. It also provides neat figures out of the box.

Try it out on [Binder](https://mybinder.org/v2/gh/moritzhuetten/dmbounds/HEAD?labpath=tutorial.ipynb):
 
 [![Screenshot](/screenshot.png)](https://mybinder.org/v2/gh/moritzhuetten/dmbounds/HEAD?labpath=tutorial.ipynb)

So far, the following experiments are included:

`CTA`, `HAWC`, `LAT`, `LHAASO`, `MAGIC`, `SWGO`, `VERITAS`, `WHIPPLE`

and channels (latex in the plot, text in the file name):

`bb`, `tautau`, `ee`, `tt`, `WW`, `gg`, Zg`,...

<img src="https://render.githubusercontent.com/render/math?math=b\bar{b},\tau^{+}\tau^{-},e^{+}e^{-},t\bar{t},W^{+}W^{-},\gamma\gamma,Z\gamma,...">

## Installation

```bash
git clone https://github.com/moritzhuetten/dmbounds.git
pip install ./dmbounds/ (-e)
```
Alternatively, do

```bash
git clone https://github.com/moritzhuetten/dmbounds.git
cd dmbounds
python setup.py install (--user)
```

### Usage

```python
from dmbounds import dmbounds as bounds
...
```

Please have a look at the [tutorial.ipynb notebook](tutorial.ipynb) for further details or [launch on Binder](https://mybinder.org/v2/gh/moritzhuetten/dmbounds/HEAD?labpath=tutorial.ipynb).

&copy; M. H&uuml;tten and M. Doro, 2022