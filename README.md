## Remove error images
python check_data.py

## Check corrupt images

```console
mogrify -set comment 'Extraneous bytes removed' *.jpg
```
