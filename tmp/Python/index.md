- # Introduction
  - [pip](pip.html)
  - [base](base.html)
  - [Encode_decode](Encode_decode.html)

- # library for Scripts
  - [argparse](argparse.html)
  - [itchat](itchat.html)
  - [json](json.html)
  - [Matplotlib](Matplotlib.html)
  - [multyprocesser](multyprocesser.html)
  - [numpy](numpy.html)
  - [OpenCV](OpenCV.html)
  - [Pandas](Pandas.html)
  - [PDF](PDF.html)
  - [PIL_np.array](PIL_np.array.html)
  - [Plot-in-the-Terminal](Plot-in-the-Terminal.html)
  - [popmail](popmail.html)
  - [pynput](pynput.html)
  - [Tensorflow](Tensorflow.html)
  - [wordcloud](wordcloud.html)

- # Biology
  - [Biopython](Biopython.html)

- # Developer
  - ## TUI
    - [TUI-libs](TUI-libs.html)
    - [npyscreen](npyscreen.html)
    - [urwid](urwid.html)
  - ## GUI
    - [kivy_Cross-platform-App](kivy_Cross-platform-App.html)
    - [kivy-Buildozer](kivy-Buildozer.html)
    - [QT](QT.html)

- # Blogs
  - [Opencv_Vedio_reverse.py](Vedio_reverse.py.html)
  - [Opencv_VedioSlice.py](VedioSlice.py.html)
  - [Opencv_Lightening-Img](Lightening-Img.html)
  - [Opencv_gif](Opencv_gif.html)
  - [Tensorflow-Numbers-k](Tensorflow-Numbers-k.html)
  - [Transfer_Learning](Transfer_Learning.html)
  - [Crawler](Crawler.html)
  - [BaseMap](BaseMap.html)
  - [HTML-server](HTML-server.html)
  - [Animation-Text](Animation-Text.html)
  - [Others](Others.html)

```bash
ls | awk '{print "["$1"]("$i")"}'| sed 's/\.md]/]/;s/\.md)/\.html)/;/yuque.yml/d;/(summary.html)/d' > P-index.md
```
