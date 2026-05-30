# data/

Lakehouse for risk inputs. **Contents of `curated/` are gitignored** — only this
README and schemas live in version control. Never commit downloaded artifacts or
proprietary P&L series.

## Layout

```
curated/   # cleaned return / P&L series ready for VaR (parquet), gitignored
```

## Expected schema (curated return series)

| column   | type             | notes                                    |
|----------|------------------|------------------------------------------|
| `date`   | date / timestamp | observation date (one row per period)    |
| `book`   | string           | book or instrument identifier            |
| `return` | float            | period return or P&L, gains positive     |

Source series are produced upstream by `mibel-derivatives` (position P&L) and
`mibel-forecasting`; drop the curated parquet here before running the VaR
notebooks.
