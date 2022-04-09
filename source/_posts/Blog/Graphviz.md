---
title: "Graphviz"
description: "Graphviz"
url: eddb3e
date: 2020/06/26
toc: true
excerpt: "Plot Nutrition Data Matrix in ggplot"
tags: [MarkDown, Graphviz]
category: [others, Blog, more]
cover: 'https://s1.ax1x.com/2020/06/26/NrHdqH.png'
thumbnail: 'https://s1.ax1x.com/2020/06/26/NrHdqH.png'
priority: 10000
---

## Graphviz

<a name="XsnEk"></a>
## 1 Quick Start
```r
digraph F {
    rankdir = LR;
    edge [style=solid];
    node [style=filled, font=Courier];

    subgraph M {
        rank = same;
        Start [label = "Lamp doesn't work", shape = box, fillcolor = "#FF0000" ];
        End   [label = "Repair lamp"      , shape = box, color = coral];

        Con1 [label = "Lamp plugged in?", shape = diamond, color = green, size = 3];
        Con2 [label = "Bulb burned out?", shape = diamond, color = green, size = 3];
    }

    subgraph C {
        rank = same;
        RB [label = "Replace bulb", shape = box, color = deepskyblue1];
        AP [label = "Plug in lamp", shape = box, color = deepskyblue1];
    }

    Start -> Con1;
    Con1 -> AP   [label = "No"];
    Con1 -> Con2 [label = "Yes"];

    Con2 -> RB  [label = "Yes"];
    Con2 -> End [label = "No"];

    AP -> End
    RB -> End
}
```

![NrHdqH.png](https://s1.ax1x.com/2020/06/26/NrHdqH.png)



```
``$`graphviz
digraph F {
    rankdir = LR;

    "node1" [ label = "{<f1> ADP |<f2> Pi }"
              shape = "record"
              ];
    "node2" [ label = "<f0> ATP| <f1> Ca"
              shape = "record"
              ];


    node2:f0 -> node1:f1
}
``$`
```

==remove the "$" from code above==

```graphviz
digraph F {
    rankdir = LR;

    "node1" [ label = "{<f1> ADP |<f2> Pi}"
              shape = "record"
              ];
    "node2" [ label = "<f0> ATP| <f1> Ca"
              shape = "record"
              ];

    node2:f0 -> node1:f1
}
```



### More exemples:
[http://graphviz.org/gallery/](http://graphviz.org/gallery/)

<a name="gPUU1"></a>
## 2 Parameters

<a name="rv6r4"></a>
### 2.1 Structure
Source:[https://graphviz.gitlab.io/_pages/doc/info/lang.html](https://graphviz.gitlab.io/_pages/doc/info/lang.html)
```r
graph  :  [ strict ] (graph | digraph) [ ID ] '{' stmt_list '}'
stmt_list  :  [ stmt [ ';' ] stmt_list ]
stmt  :  node_stmt
|  edge_stmt
|  attr_stmt
|  ID '=' ID
|  subgraph
attr_stmt  :  (graph | node | edge) attr_list
attr_list  :  '[' [ a_list ] ']' [ attr_list ]
a_list  :  ID '=' ID [ (';' | ',') ] [ a_list ]
edge_stmt  :  (node_id | subgraph) edgeRHS [ attr_list ]
edgeRHS  :  edgeop (node_id | subgraph) [ edgeRHS ]
node_stmt  :  node_id [ attr_list ]
node_id  :  ID [ port ]
port  :  ':' ID [ ':' compass_pt ]
|  ':' compass_pt
subgraph  :  [ subgraph [ ID ] ] '{' stmt_list '}'
compass_pt  :  (n | ne | e | se | s | sw | w | nw | c | _)
```


A -> {B C}<br />is equivalent to<br />A -> B<br />A -> C

<a name="pYvmH"></a>
### color for edge

![NrHtxO.png](https://s1.ax1x.com/2020/06/26/NrHtxO.png)



<a name="UebHR"></a>
## Official Documentation
Source: [http://graphviz.org/documentation/](http://graphviz.org/documentation/)


### Nodes,edges ...
Souce: [https://graphviz.gitlab.io/_pages/doc/info/attrs.html](https://graphviz.gitlab.io/_pages/doc/info/attrs.html)

Node shapes

Source: [https://graphviz.gitlab.io/_pages/doc/info/shapes.html](https://graphviz.gitlab.io/_pages/doc/info/shapes.html)
![Graphviz node shape](https://s1.ax1x.com/2020/06/26/NrHUMD.png)

### Edge shapes
```
digraph F {
  rankdir ="LR"
  A -> B [arrowhead = "box", color="green"]
  A -> C [arrowhead = "crow", style = "dotted"]
  A -> D [arrowhead = "dot", weight = 10]
  A -> E [arrowhead = "diamond", headport="s"]
  A -> F [arrowhead = "none"]
  A -> G [arrowhead = "icurve"]
  subgraph cluster_0{
    B
    C
    D
    style=filled;
		color=lightgrey;
		node [style=filled,color=white];
		label = "process #1";
  }
}
```
```graphviz
digraph F {
  rankdir ="LR"
  A -> B [arrowhead = "box", color="green"]
  A -> C [arrowhead = "crow", style = "dotted"]
  A -> D [arrowhead = "dot", weight = 10]
  A -> E [arrowhead = "diamond", headport="s"]
  A -> F [arrowhead = "none"]
  A -> G [arrowhead = "icurve"]
  subgraph cluster_0{
    B
    C
    D
    A
    style=filled;
		color=lightgrey;
		node [style=filled,color=white];
		label = "process #1";
  }

}
```

Source: [https://graphviz.gitlab.io/_pages/doc/info/arrows.html](https://graphviz.gitlab.io/_pages/doc/info/arrows.html)
![Graphviz arrowhead](https://s1.ax1x.com/2020/06/26/NrHase.png)


### Color

Source: [https://graphviz.gitlab.io/_pages/doc/info/colors.html](https://graphviz.gitlab.io/_pages/doc/info/colors.html)
![Graphviz Color](https://s1.ax1x.com/2020/06/26/NrH0Zd.png)

## Edge

### headport / tailport


`A -> n [headport = "n"]`
headport = "n","ne","e","se","s","sw","w","nw","c","_".

```graphviz
digraph{

  A -> B [headport = ""]
  A -> n [headport = "n"]
  A -> ne [headport = "ne"]
  A -> e [headport = "e"]
  A -> se [headport = "se"]
  A -> s [headport = "s"]
  e -> C
  C -> sw [headport = "sw"]
  C -> w [headport = "w"]
  C -> nw [headport = "nw"]
  C -> c [headport = "c"]
  C -> _ [headport = "_"]
  C -> D [headport = "", weight=2]

}
```

### Style

` A -> dashed [ style = "dashed"]`
style = "dashed", "dotted", "solid", "invis", "bold", "tapered"


```graphviz
digraph{

  subgraph A{
    rank = same
    A
    B
  }
  A -> dashed [ style = "dashed"]
  A -> dotted [ style = "dotted"]
  A -> solid [ style = "solid"]
  B -> invis [ style = "invis"]
  B -> bold [ style = "bold"]
  B -> tapered [ style = "tapered"]
}
```

### Multi-color
`a -> b [dir=both color="red:blue"]`
`c -> d [dir=none color="green:red;0.25:blue"]`
```graphviz
digraph G {
  a -> b [dir=both color="red:blue"]
  c -> d [dir=none color="green:red;0.25:blue"]
}
```

## Cluster

```
style=filled;
color=lightgrey;
node [style=filled,color=white];
label = "This is a cluster";
```

```graphviz
digraph{
  subgraph cluster_0{
    B
    C
    D
    A
    style=filled;
		color=lightgrey;
		node [style=filled,color=white];
		label = "This is a cluster";
  }
}
```

```graphviz
digraph{
  subgraph cluster_0{
    A
    area = 1
		label = "color = red\nfontcolor = blue";
    fontcolor = "blue";
    color = "red"
  }
  subgraph cluster_1{
    B
    bgcolor = red
		label = "bgcolor = red;\n fontname = SimHei\n fontsize = 20";
    fontname = "SimHei"
    fontsize = 20
  }
  subgraph cluster_2{
    C
    fillcolor = "blue:red"
    style=filled
    gradientangle = 45
    label = "style=filled;\nfillcolor = blue:red;\ngradientangle = 45";
  }
}

```

```graphviz
digraph{
  subgraph cluster_0{
    A -> AAAAAA
		label = "labeljust=r\n labelloc=c";
    labeljust=r
    margin = 2
  }
  subgraph cluster_1{
    B -> BBBBBB
		label = "labeljust=l\n labelloc=b";
    labeljust=l
    labelloc=b
  }
}
```

## Model

```graphviz
digraph{
  A -> B -> C -> D -> E -> F -> A
  model="circuit"
}
```
```graphviz
digraph{
  model="subset"
  A -> B -> C -> D -> E -> F -> A
}
```

```graphviz
digraph G  {
	layout=neato
	size="7,10"
	page="8.5,11"
	center=""
	node[width=.25,height=.375,fontsize=9]
	fcfpr1_1_2t_17 -> 341411;
	fcfpr1_1t_1 -> 341411;
	rdlfpr2_0_rdlt_4 -> 341411;
	fpfpr1_0_1t_1 -> 341411;
	fpfpr1_1_2t_11 -> 341411;
	rtafpr1_1_2t_28 -> 341411;
	rtafpr1_1_3t_6 -> 341411;
	rdlfpr1_1t_1 -> 358866;
	rtafpr1_1_3t_6 -> 358866;
	tmfpr1_1_3t_5 -> 358930;
	fcfpr1_1_3t_9 -> 358930;
	pcfpr1_1_3t_7 -> 358930;
	fpfpr1_1_3g_1 -> 358930;
	fpfpr1_1_3t_1 -> 358930;
	aufpr1_1_3t_1 -> 358930;
	rtafpr1_0_3g_1 -> 358930;
	rtafpr1_1_3t_6 -> 358930;
	msgfpr1_1_1g_12 -> 371943;
	rtafpr1_1_1g_8 -> 371943;
	rtafpr1_1_1t_35 -> 371943;
	rtafpr1_1_1t_45 -> 371943;
	rtafpr1_1_3t_6 -> 371943;
	tlfpr2_0_rdlg_2 -> 374300;
	fcfpr1_1_3t_8 -> 374300;
	fcfpr1_1_3t_9 -> 374300;
	rtafpr1_1_3t_6 -> 374300;
	fcfpr1_0_5g_1 -> 371942;
	fcfpr1_1_1t_19 -> 371942;
	fcfpr1_1_3t_9 -> 371942;
	fcfpr1_1_3t_9 -> 374700;
	tymsgfpr1_1_3t_3 -> 374700;
	fpfpr1_1_3t_1 -> 374700;
	rtafpr1_1_3t_7 -> 374700;
	fcfpr1_1_3g_2 -> 374741;
	fcfpr1_1_3t_9 -> 374741;
	fpfpr1_1_3t_1 -> 374741;
	rtafpr1_1_3t_7 -> 374741;
	fcfpr1_1_1t_18 -> 374886;
	fcfpr1_1_3t_9 -> 374886;
	fpfpr1_1_3t_1 -> 374886;
	rtafpr1_1_3t_7 -> 374886;
	fcfpr1_1_3t_9 -> 375039;
	fpfpr1_1_3t_1 -> 375039;
	fcfpr1_1_3t_42 -> 375507;
	fcfpr1_1_3t_9 -> 375507;
	rdlfpr2_0_rdlt_158 -> 375507;
	rtafpr1_1_3t_7 -> 375507;
	rtafpr1_1_3t_71 -> 375507;
	dbfpr1_1_3t_2 -> 375507;
	fcfpr1_1_3t_9 -> 375508;
	rdlfpr1_1g_13 -> 375508;
	rtafpr1_1_3t_7 -> 375508;
	rtafpr2_1_rdlg_1 -> 375508;
	dbfpr1_1_3t_2 -> 375508;
	fcfpr1_1_3t_9 -> 375519;
	fpfpr1_1_3g_1 -> 375519;
	fpfpr1_1_3t_1 -> 375519;
	fcfpr1_1_3t_9 -> 377380;
	rdlfpr1_1g_16 -> 377380;
	rdlfpr1_1t_100 -> 377380;
	fcfpr1_0_2g_1 -> 377719;
	fcfpr1_1_3t_10 -> 377719;
	fcfpr1_1_3t_7 -> 377719;
	fcfpr1_1_3t_9 -> 377719;
	rdlfpr2_0_rdlg_12 -> 377719;
	rdlfpr2_0_rdlt_108 -> 377719;
	rdlfpr2_0_rdlt_27 -> 377719;
	rdlfpr2_0_rdlt_30 -> 377719;
	fcfpr1_1_3t_9 -> 377763;
	fcfpr1_1_3t_9 -> 379848;
	fpfpr1_1_3t_1 -> 379848;
	fcfpr1_1_3t_9 -> 380571;
	fcfpr1_1_3t_9 -> 380604;
	fpfpr1_1_3t_1 -> 380604;
	fcfpr1_1_3t_9 -> 381211;
	fpfpr1_1_3t_1 -> 381211;
	fcfpr1_1_3t_9 -> 381835;
	fcfpr1_1_3t_9 -> 381897;
	fcfpr1_1_3t_9 -> 381901;
	fpfpr1_1_3t_1 -> 381901;
	fcfpr1_1_3t_9 -> 382103;
	rtafpr1_1_3t_7 -> 382103;
	fcfpr1_1_3t_9 -> 382161;
	fcfpr1_1_3t_9 -> 383174;
	fpfpr1_1_3t_1 -> 383174;
	rtafpr1_1_3t_7 -> 383174;
	fpfpr1_1_3g_1 -> 352010;
	fpfpr1_1_3t_1 -> 352010;
	fpfpr1_1_3t_1 -> 382409;
	fpfpr1_1_3t_1 -> 382827;
	fpfpr1_1_3t_1 -> 382928;
	rtafpr1_1_3t_7 -> 382928;
	tlfpr1_1_1t_5 -> 358224;
	tymsgfpr1_1_1t_23 -> 358224;
	tymsgfpr1_1_3t_3 -> 358224;
	rcfpr0_0_1t_9 -> 358224;
	rcfpr1_1_1t_5 -> 358224;
	odfpr0_0_1t_8 -> 358224;
	odfpr1_1_1t_6 -> 358224;
	ecdsgfpr1_1_1t_4 -> 358224;
	tymsgfpr1_1_1t_18 -> 358900;
	tymsgfpr1_1_3t_3 -> 358900;
	rcfpr1_1_1t_100 -> 358900;
	rcfpr1_1_1t_22 -> 358900;
	rcfpr1_1_1t_37 -> 358900;
	odfpr1_1_1t_21 -> 358900;
	tymsgfpr1_1_3t_3 -> 372568;
	rcfpr1_1_1t_30 -> 372568;
	odfpr1_1_1t_31 -> 372568;
	tlfpr1_1_1t_20 -> 375557;
	tymsgfpr1_1_1t_24 -> 375557;
	tymsgfpr1_1_3t_3 -> 375557;
	rcfpr1_1_1t_11 -> 375557;
	odfpr1_1_1t_9 -> 375557;
	ecdsgfpr1_1_1t_19 -> 375557;
	rtafpr1_1_1g_14 -> 376956;
	rtafpr1_1_1t_64 -> 376956;
	rtafpr1_1_2t_18 -> 376956;
	rtafpr1_1_3t_30 -> 376956;
	rtafpr1_1_3t_7 -> 376956;
	rtafpr1_1_3t_7 -> 379339;
	rtafpr1_1_1t_14 -> 379422;
	rtafpr1_1_1t_20 -> 379422;
	rtafpr1_1_3t_7 -> 379422;
	rtafpr1_1_3t_7 -> 383039;
	fcfpr1_1_1t_18 -> 359471;
	fcfpr2_0_1t_1 -> 359471;
	fcfpr2_0_1t_2 -> 359471;
	ccsfpr2_0_1t_99 -> 359471;
	fcfpr1_1_3t_42 -> 384096;
	rtafpr1_1_3t_71 -> 384096;
	tlfpr1_0_4g_4 -> 354290;
	rcfpr0_0_1t_9 -> 354290;
	odfpr0_0_1t_8 -> 354290;
	pagfpr1_1_1t_23 -> 354290;
	rcfpr1_1_1t_5 -> 379864;
	rcfpr1_1_1t_100 -> 382574;
	rcfpr1_1_1t_22 -> 382574;
	rcfpr1_1_1t_37 -> 382574;
	rcfpr1_1_1t_30 -> 370706;
	rcfpr1_1_1t_30 -> 377908;
	rcfpr1_1_1t_30 -> 377924;
	rcfpr1_1_1t_30 -> 377971;
	rcfpr1_1_1t_30 -> 377980;
	odfpr1_1_1t_31 -> 377980;
	rcfpr1_1_1t_30 -> 378362;
	rcfpr1_1_1t_30 -> 378656;
	rcfpr1_1_1t_30 -> 378666;
	rcfpr1_1_1t_30 -> 379169;
	odfpr1_1_1t_31 -> 379169;
	rcfpr1_1_1t_110 -> 379341;
	rcfpr1_1_1t_30 -> 379341;
	rcfpr1_1_1t_62 -> 379341;
	odfpr1_1_1t_31 -> 379341;
	rcfpr1_1_1t_30 -> 379972;
	rcfpr1_1_1t_30 -> 380298;
	rcfpr1_1_1t_30 -> 380448;
	rcfpr1_1_1t_30 -> 380475;
	odfpr1_1_1t_31 -> 380475;
	rcfpr1_1_1t_30 -> 380526;
	odfpr1_1_1t_31 -> 357430;
	rcfpr1_1_1t_11 -> 379968;
	odfpr1_1_1t_9 -> 379968;
	ccsfpr2_0_1t_99 -> 359100;
	ccsfpr2_0_1t_99 -> 376529;
	ccsfpr2_0_1t_99 -> 377801;
	ccsfpr2_0_1t_99 -> 379126;
	ccsfpr2_0_1t_99 -> 379212;
	ccsfpr2_0_1t_99 -> 380285;
	ccsfpr2_0_1t_99 -> 380963;
	ccsfpr2_0_1t_99 -> 384909;
	tlfpr1_0_4g_4 -> 358471;
	odfpr0_0_1t_7 -> 358471;
	odfpr1_0_1t_36 -> 358471;
	odfpr1_0_3t_18 -> 358471;
	odfpr1_0_3t_21 -> 358471;
	tlfpr1_0_4g_4 -> 375024;
	tlfpr1_0_4g_4 -> 375027;
	rcfpr1_1_1t_110 -> 381710;
	rcfpr1_1_1t_62 -> 381710;
	rcfpr1_1_1t_110 -> 381775;
	rcfpr1_1_1t_62 -> 381775;
	rcfpr1_1_1t_110 -> 382436;
	fcfpr1_1_3t_34 -> 382528;
	rcfpr1_1_1t_110 -> 382528;
	rtafpr1_1_3t_48 -> 382528;
	rcfpr1_1_1t_110 -> 382566;
	rcfpr1_1_1t_110 -> 382572;
	odfpr0_0_1t_7 -> 353506;
	rcfpr1_0_1t_35 -> 370509;
	odfpr0_0_1t_7 -> 370509;
	odfpr0_0_1t_7 -> 370510;
	odfpr1_0_1t_38 -> 370510;
	tlfpr1_0_4g_5 -> 354546;
	rcfpr1_1_1t_61 -> 354546;
	odfpr1_0_3t_18 -> 354546;
	odfpr1_0_3t_20 -> 354546;
	odfpr1_0_3t_18 -> 354757;
	odfpr1_0_3t_20 -> 354757;
	odfpr1_0_3t_18 -> 354766;
	odfpr1_0_3t_20 -> 354766;
	odfpr1_0_3t_18 -> 354771;
	odfpr1_0_3t_20 -> 354771;
	odfpr1_0_3t_18 -> 354785;
	odfpr1_0_3t_23 -> 354785;
	odfpr1_0_3t_24 -> 354785;
	odfpr1_0_3t_18 -> 354878;
	odfpr1_0_3t_23 -> 354878;
	odfpr1_0_3t_24 -> 354878;
	odfpr1_0_3t_18 -> 355080;
	odfpr1_0_3t_23 -> 355080;
	odfpr1_0_3t_24 -> 355080;
	odfpr1_0_3t_18 -> 355288;
	odfpr1_0_3t_23 -> 355288;
	odfpr1_0_3t_24 -> 355288;
	odfpr2_0_03t_13 -> 355288;
	odfpr1_0_3t_18 -> 355800;
	odfpr1_0_3t_21 -> 355800;
	odfpr1_0_3t_18 -> 356116;
	odfpr1_0_3t_21 -> 356116;
	odfpr1_0_3t_18 -> 356741;
	odfpr1_0_3t_21 -> 356741;
	odfpr1_0_3t_18 -> 357340;
	odfpr1_0_3t_21 -> 357340;
	odfpr1_0_3t_18 -> 357538;
	odfpr1_0_3t_21 -> 357538;
	odfpr1_0_3t_18 -> 357769;
	odfpr1_0_3t_21 -> 357769;
	odfpr1_0_3t_18 -> 357793;
	odfpr1_0_3t_21 -> 357793;
	odfpr1_0_3t_18 -> 358155;
	odfpr1_0_3t_21 -> 358155;
	odfpr1_0_3t_18 -> 358157;
	odfpr1_0_3t_21 -> 358157;
	odfpr1_0_3t_18 -> 358159;
	odfpr1_0_3t_21 -> 358159;
	odfpr1_0_3t_18 -> 358584;
	odfpr1_0_3t_21 -> 358584;
	odfpr1_0_3t_18 -> 360104;
	odfpr1_0_3t_21 -> 360104;
	odfpr1_0_3t_18 -> 360144;
	odfpr1_0_3t_21 -> 360144;
	odfpr1_0_3t_18 -> 360672;
	odfpr1_0_3t_21 -> 360672;
	odfpr1_0_3t_5 -> 360672;
	odfpr1_0_3t_18 -> 360839;
	odfpr1_0_3t_21 -> 360839;
	odfpr1_0_3t_18 -> 371187;
	tlfpr1_0_3g_5 -> 373300;
	odfpr1_0_3t_12 -> 373300;
	odfpr1_0_3t_18 -> 373300;
	odfpr1_0_3t_18 -> 375134;
	odfpr1_0_5t_18 -> 375134;
	rcfpr0_0_1t_10 -> 375319;
	odfpr1_0_3t_18 -> 375319;
	odfpr1_0_3t_36 -> 375319;
	odfpr1_0_5t_17 -> 375319;
	odfpr1_0_5t_19 -> 375319;
	odfpr1_0_3t_18 -> 375499;
	odfpr1_0_3t_18 -> 377220;
	odfpr1_0_5t_21 -> 377220;
	tlfpr1_0_3g_7 -> 377562;
	tlfpr1_1_1t_3 -> 377562;
	odfpr1_0_3t_18 -> 377562;
	odfpr1_0_3t_36 -> 377562;
	odfpr1_0_5t_20 -> 377562;
	odfpr1_0_3t_18 -> 378108;
	odfpr1_0_3t_6 -> 378108;
	odfpr1_0_5t_20 -> 354221;

	odfpr0_0_1t_7 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tlfpr1_0_3g_5 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr0_0_1t_8 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_1_1t_61 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1t_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_3t_18 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tlfpr1_0_3g_7 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_1_1t_62 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	ccsfpr2_0_1t_99 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tymsgfpr1_1_3t_3 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr0_0_1t_9 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_1t_14 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_3t_30 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_1_1t_110 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	dbfpr1_1_3t_2 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_1g_8 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_1_1t_30 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tlfpr1_1_1t_20 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_1t_64 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tlfpr2_0_rdlg_2 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_2t_28 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tlfpr1_1_1t_3 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_1_1t_6 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fpfpr1_1_3t_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	aufpr1_1_3t_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_3t_34 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_1_1t_5 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_1t_18 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_3t_36 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tlfpr1_1_1t_5 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_1t_19 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_1_1t_9 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_3t_7 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_1_1t_37 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_3t_8 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_1_1t_21 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_3t_9 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr2_0_rdlt_27 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_3g_2 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_1t_35 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_5t_20 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fpfpr1_1_3g_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_5t_21 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fpfpr1_1_2t_11 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	ecdsgfpr1_1_1t_19 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_1t_36 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_1g_14 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tymsgfpr1_1_1t_23 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tymsgfpr1_1_1t_24 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_1t_38 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_0_2g_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr1_1t_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr0_0_1t_10 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_1_1t_100 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr2_0_rdlt_108 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	pcfpr1_1_3t_7 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_3t_20 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	ecdsgfpr1_1_1t_4 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tmfpr1_1_3t_5 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_3t_21 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fpfpr1_0_1t_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_3t_23 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_1_1t_22 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	pagfpr1_1_1t_23 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_3t_71 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_2t_18 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr2_0_rdlt_158 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_3t_6 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_3t_24 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_3t_7 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_0_3g_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_1t_20 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr1_1g_13 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_0_1t_35 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_2t_17 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr2_1_rdlg_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr2_0_rdlt_4 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr1_1g_16 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr2_0_1t_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr2_0_1t_2 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr1_1t_100 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	msgfpr1_1_1g_12 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr2_0_rdlt_30 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_3t_5 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tlfpr1_0_4g_4 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_3t_42 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_3t_6 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tlfpr1_0_4g_5 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_3t_48 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_5t_17 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_5t_18 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	tymsgfpr1_1_1t_18 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_5t_19 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_1_3t_10 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	fcfpr1_0_5g_1 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_0_3t_12 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr2_0_03t_13 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rcfpr1_1_1t_11 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	odfpr1_1_1t_31 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rdlfpr2_0_rdlg_12 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
	rtafpr1_1_1t_45 [label="",shape=circle,height=0.12,width=0.12,fontsize=1];
}
```

## For more
please visit: 

使用graphviz绘制流程图（2015版）<br />[http://icodeit.org/2015/11/using-graphviz-drawing/](http://icodeit.org/2015/11/using-graphviz-drawing/)