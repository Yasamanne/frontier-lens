from __future__ import annotations

from frontier_lens.templates.spec import ItemSpec, python_stem

TOPIC = "constraint_counting"


def _count_code(alphabet: str, length: int, extra: str) -> str:
    return (
        "from itertools import product\n"
        f"alpha = {alphabet!r}\n"
        f"n = 0\n"
        f"for cand in product(alpha, repeat={length}):\n"
        f"    s = ''.join(cand)\n"
        f"    {extra}\n"
        "print(n)\n"
    )


def _item(level: int, name: str, stem: str, alphabet: str, length: int, extra: str, **params) -> ItemSpec:
    code = _count_code(alphabet, length, extra)
    return ItemSpec(
        topic=TOPIC,
        level=level,
        template=name,
        params={"alphabet": alphabet, "length": length, **params},
        stem=python_stem(code, stem),
        code=code,
        lang="python",
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    items = [
        _item(1, "all_strings", "how many 3-letter strings over {A,B}?", "AB", 3, "n += 1"),
        _item(1, "binary_len2", "how many 2-letter strings over {0,1}?", "01", 2, "n += 1"),
    ]
    l2_extra = [
        ("5 letters over {A,B,C}, no two adjacent equal, first equals last.", "ABC", 5,
         "if all(s[i] != s[i+1] for i in range(len(s)-1)) and s[0]==s[-1]:\n        n += 1"),
        ("4 letters over {A,B}, no adjacent repeats.", "AB", 4,
         "if all(s[i] != s[i+1] for i in range(len(s)-1)):\n        n += 1"),
        ("3 letters over {A,B,C} containing at least one A.", "ABC", 3,
         "if 'A' in s:\n        n += 1"),
        ("4 letters over {A,B,C} with exactly two A's.", "ABC", 4,
         "if s.count('A')==2:\n        n += 1"),
        ("3 letters over {A,B,C} that are palindromes.", "ABC", 3,
         "if s==s[::-1]:\n        n += 1"),
        ("5 letters over {A,B} with no three A's in a row.", "AB", 5,
         "if 'AAA' not in s:\n        n += 1"),
        ("4 letters over {X,Y,Z} starting with X.", "XYZ", 4,
         "if s[0]=='X':\n        n += 1"),
        ("3 letters over {A,B,C} with all letters distinct.", "ABC", 3,
         "if len(set(s))==3:\n        n += 1"),
        ("4 letters over {A,B} ending with B.", "AB", 4,
         "if s[-1]=='B':\n        n += 1"),
        ("3 letters over {A,B,C} with more A's than B's.", "ABC", 3,
         "if s.count('A')>s.count('B'):\n        n += 1"),
        ("4 letters over {A,B,C} sorted (nondecreasing).", "ABC", 4,
         "if list(s)==sorted(s):\n        n += 1"),
        ("5 letters over {A,B} with exactly 3 A's.", "AB", 5,
         "if s.count('A')==3:\n        n += 1"),
    ]
    for i, (stem, alpha, n, extra) in enumerate(l2_extra):
        items.append(_item(2, "count_l2", stem, alpha, n, extra, idx=i))
    l3_extra = [
        ("6 letters over {A,B,C}, no adjacent repeats, exactly two A's, not starting with C.",
         "ABC", 6,
         "if all(s[i]!=s[i+1] for i in range(5)) and s.count('A')==2 and s[0]!='C':\n        n += 1"),
        ("5 letters over {A,B,C}, palindrome, at least one B.",
         "ABC", 5,
         "if s==s[::-1] and 'B' in s:\n        n += 1"),
        ("6 letters over {A,B}, no four B's in a row, starts and ends with A.",
         "AB", 6,
         "if s[0]=='A' and s[-1]=='A' and 'BBBB' not in s:\n        n += 1"),
        ("4 letters over {A,B,C,D} with all distinct and first < last alphabetically.",
         "ABCD", 4,
         "if len(set(s))==4 and s[0]<s[-1]:\n        n += 1"),
        ("5 letters over {A,B,C}, exactly one pair of adjacent equals, rest strict.",
         "ABC", 5,
         "adj=sum(s[i]==s[i+1] for i in range(4))\n    if adj==1:\n        n += 1"),
        ("6 letters over {A,B,C} with even number of A's and no leading B.",
         "ABC", 6,
         "if s.count('A')%2==0 and s[0]!='B':\n        n += 1"),
        ("5 letters over {A,B}, more A's than B's, no AA at start.",
         "AB", 5,
         "if s.count('A')>s.count('B') and not s.startswith('AA'):\n        n += 1"),
        ("4 letters over {A,B,C} that contain ABC as a subsequence (not necessarily contiguous).",
         "ABC", 4,
         "i=0\n    for ch in s:\n        if i<3 and ch=='ABC'[i]:\n            i+=1\n    if i==3:\n        n += 1"),
        ("5 letters over {A,B,C} with first=last and no CCC.",
         "ABC", 5,
         "if s[0]==s[-1] and 'CCC' not in s:\n        n += 1"),
        ("6 letters over {A,B} with exactly 2 runs (maximal equal blocks).",
         "AB", 6,
         "runs=1+sum(s[i]!=s[i-1] for i in range(1,6))\n    if runs==2:\n        n += 1"),
        ("4 letters over {A,B,C} whose sorted unique letters have length 2.",
         "ABC", 4,
         "if len(set(s))==2:\n        n += 1"),
        ("5 letters over {A,B,C}, not a palindrome, exactly two C's.",
         "ABC", 5,
         "if s!=s[::-1] and s.count('C')==2:\n        n += 1"),
    ]
    for i, (stem, alpha, n, extra) in enumerate(l3_extra):
        items.append(_item(3, "count_l3", stem, alpha, n, extra, idx=i))
    return items
