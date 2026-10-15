from __future__ import annotations

from frontier_lens.templates.spec import ItemSpec, shell_stem

TOPIC = "shell_pipelines"


def _item(level: int, name: str, files: dict[str, str], command: str, question: str, **params) -> ItemSpec:
    return ItemSpec(
        topic=TOPIC,
        level=level,
        template=name,
        params=params,
        stem=shell_stem(files, command, question),
        code=command,
        lang="shell",
        input_files=files,
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    items: list[ItemSpec] = []
    colors = {"colors.txt": "red\nblue\nred\ngreen\n"}
    items.append(
        _item(1, "uniq_sort_count", colors, "sort -u colors.txt | wc -l", "what number does this print?", n=4)
    )
    items.append(
        _item(
            1,
            "line_count",
            {"notes.txt": "a\nb\nc\n"},
            "cat notes.txt | wc -l",
            "what number does this print?",
        )
    )

    csvs = [
        "id,name,score\n1,ann,10\n2,bob,7\n3,cy,10\n4,dee,3\n",
        "id,name,score\n1,ann,2\n2,bob,2\n3,cy,9\n4,dee,9\n5,ed,1\n",
        "sku,color,qty\na,red,1\nb,red,2\nc,blue,2\nd,green,0\n",
        "sku,color,qty\na,red,5\nb,blue,5\nc,red,1\n",
        "city,temp\naus,30\ndal,30\nhou,28\naus,31\n",
        "city,temp\nnyc,10\nchi,10\nla,20\nsea,10\n",
        "user,role\na,admin\nb,user\nc,user\nd,admin\ne,user\n",
        "user,role\na,ops\nb,ops\nc,dev\n",
        "code,tag\n1,x\n2,y\n3,x\n4,z\n5,y\n",
        "code,tag\n9,p\n8,p\n7,q\n",
        "n,v\n1,aa\n2,ab\n3,aa\n4,ac\n5,ab\n6,aa\n",
        "n,v\n1,t\n2,u\n3,t\n4,t\n",
    ]
    for i, body in enumerate(csvs):
        fname = "data.csv"
        files = {fname: body}
        cmd = "cut -d, -f2 data.csv | uniq | wc -l"
        items.append(
            _item(2, "cut_uniq_nosort", files, cmd, "what number does this print?", idx=i, trap="uniq_without_sort")
        )

    hard_bodies = [
        ("id,name,score\n1,ann,10\n2,bob,7\n3,cy,9\n4,dee,3\n", "sort -t, -k3,3 data.csv | tail -n 1 | cut -d, -f2"),
        ("id,name,score\n1,ann,2\n2,bob,12\n3,cy,9\n", "sort -t, -k3,3 data.csv | tail -n 1 | cut -d, -f2"),
        ("id,name,score\n1,ann,8\n2,bob,8\n3,cy,11\n4,dee,1\n", "sort -t, -k3,3 data.csv | head -n 2 | tail -n 1 | cut -d, -f2"),
        ("sku,color,qty\na,red,11\nb,blue,2\nc,green,3\n", "sort -t, -k3,3 data.csv | tail -n 1 | cut -d, -f1"),
        ("sku,color,qty\na,red,4\nb,blue,4\nc,green,40\n", "sort -t, -k3,3 data.csv | tail -n 1 | cut -d, -f2"),
        ("city,temp\naus,9\ndal,30\nhou,8\n", "sort -t, -k2,2 data.csv | tail -n 1 | cut -d, -f1"),
        ("city,temp\nnyc,100\nchi,20\nla,3\n", "sort -t, -k2,2 data.csv | head -n 2 | tail -n 1 | cut -d, -f1"),
        ("user,role,n\na,admin,1\nb,user,11\nc,user,2\n", "cut -d, -f2 data.csv | sort | uniq | wc -l"),
        ("user,role,n\na,ops,3\nb,ops,12\nc,dev,4\nd,dev,1\n", "cut -d, -f2 data.csv | sort | uniq | wc -l"),
        ("code,tag\n1,x\n2,y\n3,x\n4,z\n", "cut -d, -f2 data.csv | sort | uniq -c | sort -n | tail -n 1 | grep -c x || true"),
        ("n,v\n1,aa\n2,ab\n3,aa\n4,ac\n", "cut -d, -f2 data.csv | sort | uniq | wc -l"),
        ("n,v\n1,t\n2,u\n3,t\n4,t\n5,v\n", "cut -d, -f2 data.csv | sort -u | wc -l"),
    ]
    # The grep -c item is fragile and uses || true which is banned. Replace it.
    hard_bodies[9] = (
        "code,tag\n1,x\n2,y\n3,x\n4,z\n",
        "cut -d, -f2 data.csv | sort | uniq | wc -l",
    )
    for i, (body, cmd) in enumerate(hard_bodies):
        files = {"data.csv": body}
        items.append(
            _item(
                3,
                "lexical_sort_or_uniq",
                files,
                cmd,
                "what number or token does this print?",
                idx=i,
            )
        )
        items[-1].numeric = not ("cut -d, -f" in cmd and "wc" not in cmd)
    return items
