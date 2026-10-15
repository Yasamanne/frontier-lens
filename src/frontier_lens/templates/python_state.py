from __future__ import annotations

from frontier_lens.templates.spec import ItemSpec, python_stem

TOPIC = "python_state"


def _item(level: int, name: str, code: str, question: str, **params) -> ItemSpec:
    return ItemSpec(
        topic=TOPIC,
        level=level,
        template=name,
        params=params,
        stem=python_stem(code, question),
        code=code,
        lang="python",
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    items = [
        _item(1, "append_len", "x=[1,2,3]\nx.append(4)\nprint(len(x))\n", "what number does this print?"),
        _item(1, "sum_list", "x=[2,3,5]\nprint(sum(x))\n", "what number does this print?"),
    ]
    l2 = [
        "def add(x, acc=[]):\n    acc.append(x)\n    return len(acc)\nprint(add(1)+add(2)+add(3))\n",
        "a=[1,2,3]\nb=a\nb.append(4)\nprint(len(a))\n",
        "d={'n':0}\ne=d\ne['n']=5\nprint(d['n'])\n",
        "xs=[0]\nfor _ in range(3):\n    xs.append(xs[-1]+1)\nprint(sum(xs))\n",
        "s=set([1,2,2,3])\nprint(len(s))\n",
        "t=(1,2,[3])\nt[2].append(4)\nprint(len(t[2]))\n",
        "a={'x':1}\nb={'x':1}\nprint(int(a is b))\n",
        "a={'x':1}\nb=a\nprint(int(a is b))\n",
        "vals=[1,2,3]\nprint(vals.pop()+vals.pop())\n",
        "q=[1,2,3,4]\nprint(q[1:3][0]+q[-1])\n",
        "n=0\nfor i in [1,2,3]:\n    n+=i\nprint(n)\n",
        "from collections import Counter\nprint(Counter('ababa')['a'])\n",
    ]
    for i, code in enumerate(l2):
        items.append(_item(2, "state_l2", code, "what number does this print?", idx=i))
    l3 = [
        "grid=[[0]*3]*3\ngrid[0][1]=5\nprint(sum(sum(row) for row in grid))\n",
        "def f(xs=None):\n    if xs is None:\n        xs=[]\n    xs.append(1)\n    return xs\nprint(len(f())+len(f())+len(f([0])))\n",
        "a=[1,2,3]\nfor x in list(a):\n    a.append(x)\nprint(len(a))\n",
        "d={}\nfor k in ['a','b','a']:\n    d[k]=d.get(k,0)+1\nprint(d['a']*10+d['b'])\n",
        "class Box:\n    def __init__(self, n):\n        self.n=n\nb=Box(1)\nc=b\nc.n=4\nprint(b.n)\n",
        "xs=[[], []]\nys=xs[0]\nys.append(9)\nprint(len(xs[0])+len(xs[1])*10)\n",
        "from collections import defaultdict\nd=defaultdict(int)\nfor ch in 'mississippi':\n    d[ch]+=1\nprint(d['s']+d['i'])\n",
        "n=1\nfor i in range(1,5):\n    n*=i\nprint(n)\n",
        "seen=set()\nfor w in ['aa','ab','aa','ba']:\n    seen.add(w[0])\nprint(len(seen))\n",
        "a=[0,0,0]\nb=a\na[1]=2\nprint(sum(b))\n",
        "from itertools import accumulate\nprint(list(accumulate([1,2,3,4]))[-1])\n",
        "x=2\ndef inner():\n    return x*3\nprint(inner()+x)\n",
    ]
    for i, code in enumerate(l3):
        items.append(_item(3, "state_l3", code, "what number does this print?", idx=i))
    return items
