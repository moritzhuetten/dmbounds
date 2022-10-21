# DMbounds

A database for all latest indirect Dark Matter limits, and to compare against direct and collider limits. It also provides neat figures out of the box.

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

Please have a look at the [tutorial.ipynb notebook](tutorial.ipynb) for further details.

&copy; M. H&uuml;tten and M. Doro, 2022