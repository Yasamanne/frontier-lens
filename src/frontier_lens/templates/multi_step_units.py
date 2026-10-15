from __future__ import annotations

from frontier_lens.templates.spec import ItemSpec

TOPIC = "multi_step_units"


def _py(expr: str) -> str:
    return (
        "from decimal import Decimal, ROUND_HALF_UP\n"
        f"val = {expr}\n"
        "if isinstance(val, Decimal):\n"
        "    print(val)\n"
        "else:\n"
        "    print(val)\n"
    )


def _item(level: int, name: str, stem: str, expr: str, **params) -> ItemSpec:
    return ItemSpec(
        topic=TOPIC,
        level=level,
        template=name,
        params=params,
        stem=stem + "\n\nQuestion: report the number asked. Round half up to the nearest cent when money.",
        code=_py(expr),
        lang="python",
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    items = [
        _item(1, "km_m", "A path is 2.5 km long. How many meters is that?", "Decimal('2.5')*1000"),
        _item(1, "min_sec", "A wait is 4 minutes. How many seconds?", "4*60"),
    ]
    l2 = [
        ("A tank holds 3 cubic meters. A pump adds 40 liters per minute for 10 minutes, then 1.2 liters per second for 50 seconds. How many liters were added in total?",
         "Decimal('40')*10 + Decimal('1.2')*50"),
        ("A ribbon is 2.4 m. You cut 35 cm then 12 cm. How many centimeters remain?",
         "Decimal('2.4')*100 - 35 - 12"),
        ("Walk 1.5 km at 5 km/h. How many minutes does it take?",
         "(Decimal('1.5')/Decimal('5'))*60"),
        ("A bottle is 750 mL. You drink 0.25 L. How many milliliters remain?",
         "Decimal('750') - Decimal('0.25')*1000"),
        ("Convert 36 km/h to meters per second.",
         "Decimal('36')*1000/3600"),
        ("A square field is 80 m on a side. Area in hectares? (1 ha = 10000 m^2)",
         "Decimal('80')*80/10000"),
        ("Mix 2.5 L of water with 500 mL of juice. Total milliliters?",
         "Decimal('2.5')*1000 + 500"),
        ("A car uses 6 L per 100 km. Fuel for 250 km, in liters.",
         "Decimal('6')/100*250"),
        ("3 hours 20 minutes in minutes.",
         "3*60+20"),
        ("A crate is 12.5 kg. Add 800 g. Total kilograms?",
         "Decimal('12.5') + Decimal('0.8')"),
        ("Room 4 m by 5 m, height 2.5 m. Volume in cubic meters.",
         "Decimal('4')*5*Decimal('2.5')"),
        ("Discount 15% off 80. What is the sale price?",
         "Decimal('80')*(Decimal('1')-Decimal('0.15'))"),
    ]
    for i, (stem, expr) in enumerate(l2):
        items.append(_item(2, "units_l2", stem, expr, idx=i))
    l3 = [
        ("A trip is 340 miles. Fuel use is 6.5 L per 100 km. Fuel costs $1.80 per liter. There is also a 60 L tank mentioned that is not used. 1 mile = 1.60934 km. Cost in dollars to the nearest cent, rounding half up.",
         "(Decimal('340')*Decimal('1.60934')/Decimal('100')*Decimal('6.5')*Decimal('1.80')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)"),
        ("Pipe fills 18 L/min for 4 min, then 0.4 L/s for 90 s, then you drain 1.5 L. Net liters added.",
         "Decimal('18')*4 + Decimal('0.4')*90 - Decimal('1.5')"),
        ("A recipe needs 250 g flour per loaf. You bake 7 loaves and already have 0.9 kg. Additional grams of flour needed.",
         "Decimal('250')*7 - Decimal('0.9')*1000"),
        ("Speed 90 km/h for 40 minutes, then 20 m/s for 3 minutes. Total kilometers. Round half up to 2 decimal places.",
         "(Decimal('90')*Decimal('40')/60 + Decimal('20')*3*60/1000).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)"),
        ("A tank 2 m × 1.5 m × 80 cm (height). Capacity in liters (1 m^3 = 1000 L).",
         "Decimal('2')*Decimal('1.5')*Decimal('0.80')*1000"),
        ("Convert 72 F to C: (F-32)*5/9. Round half up to 2 decimal places.",
         "((Decimal('72')-32)*5/9).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)"),
        ("Wire 12.5 m. Cut 8 pieces of 30 cm and one of 1.2 m. Remaining centimeters.",
         "Decimal('12.5')*100 - 8*30 - Decimal('1.2')*100"),
        ("Payroll $18.50/h for 37.5 hours plus $27 overtime for 4 hours. Total dollars to the cent, half up.",
         "(Decimal('18.50')*Decimal('37.5') + Decimal('27')*4).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)"),
        ("Map scale 1:25000. 8.4 cm on the map. Real distance in kilometers. Half up to 2 decimals.",
         "(Decimal('8.4')/100*25000/1000).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)"),
        ("Battery 4.8 kWh. Device draws 120 W. Hours until empty.",
         "Decimal('4.8')*1000/120"),
        ("Two rooms 3.2 m × 4 m and 2.5 m × 5.5 m. Total area m^2. Half up to 1 decimal.",
         "(Decimal('3.2')*4 + Decimal('2.5')*Decimal('5.5')).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)"),
        ("Currency: 50 USD at 0.92 EUR per USD, 2% fee subtracted after conversion. EUR to the cent, half up.",
         "(Decimal('50')*Decimal('0.92')*(Decimal('1')-Decimal('0.02'))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)"),
    ]
    for i, (stem, expr) in enumerate(l3):
        items.append(_item(3, "units_l3", stem, expr, idx=i))
    return items
