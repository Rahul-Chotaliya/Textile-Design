# Saree Processor

## Installation

```bash
pip install -r requirements.txt
```

## Single File

```bash
python main.py sketch.bmp output.bmp
```

## Batch Processing

```bash
python main.py --batch /sketches/ /outputs/
```

## Output Palette

| Color      | RGB         | Usage              |
|------------|-------------|--------------------|
| Cream      | (245,240,225) | Background        |
| Burgundy   | (90,20,50)  | Design Lines       |
| Gold       | (195,145,30)| Fill Regions       |

## Constraints Satisfied

- Output size = Input size exactly
- Exactly 2-3 colors from fixed palette
- No symmetry operations
- All processing mask-based (no direct RGB editing)
- Background detected as largest connected border region
- Noise removal threshold dynamic (based on image size)
- All contours remain continuous
- Gaps < 3px closed before region filling
- Region filling only inside closed contours
- No interpolation, no resizing, no blurring after segmentation
- Edge sharpness preserved
- Structure deviation < 3%
- Input/output BMP only
- Pixel mapping deterministic
