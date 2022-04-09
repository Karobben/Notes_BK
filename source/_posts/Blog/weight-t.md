---
toc: true
url: weight_t
covercopy: © Karobben
priority: 10000
date: 2021-09-21 18:31:55
title: "Weight Tracking"
ytitle: "=="
description: "Echart graphics"
excerpt: "No fancying staff, just echarts"
tags: []
category: []
cover: ""
thumbnail: ""
---

# Tracking

{% echarts 400 '85%' %}
option = {
  title: {
    text: 'Weight Tracking'
  },
  tooltip: {
    trigger: 'axis'
  },
  legend: {
    data: ['Ken', 'YR']
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true
  },

  xAxis: {
    type: 'time',
    boundaryGap: false
    },
  yAxis: {
    type: 'value',
    min: 52,
  },
  series: [
    {
      name: 'Ken',
      type: 'line',
      smooth: 'true',
      data: [["2021-09-04", 56.4],
      ["2021-09-21", 55.8],
      ["2021-09-22", 54.6],
      ["2021-09-23", 55.3],
      ["2021-09-24", 54.6],
      ["2021-09-25", 54.6],
      ["2021-09-26", 55.0],
      ["2021-09-27", 54.7],
      ["2021-09-28", 54.0],
      ["2021-09-29", 54.0],
      ["2021-09-30", 54.4],
      ["2021-10-01", 54.1],
      ["2021-10-02", 54.4],
      ["2021-10-03", 54.4],
      ["2021-10-04", 54.4],
      ["2021-10-05", 54.2],
      ["2021-10-06", 54.4],
      ["2021-10-07", 54.7],
      ["2021-10-07", 54.8],
      ["2021-10-09", 54.5],
      ["2021-10-10", 54.2],
      ["2021-10-11", 54.4],
      ["2021-10-12", 54.4],
      ["2021-10-13", 54.4],
      ["2021-10-14", 54.4],
      ["2021-10-15", 54.4],
      ["2021-10-16", 54.6],      
      ["2021-10-17", 55.0],
      ["2021-10-18", 54.6],
      ["2021-10-19", 54.3],
      ["2021-10-20", 54.6],
      ["2021-10-21", 54.7],
      ["2021-10-22", 54.5],
      ["2021-10-23", 54.6],
      ["2021-10-24", 55.0],
      ["2021-10-25", 54.3],
      ["2021-10-26", 54.3],
      ["2021-10-27", 54.0],
      ["2021-10-28", 54.0],
      ["2021-10-29", 54.1],
      ["2021-10-30", 54.3],
      ["2021-10-31", 54.5],
      ["2021-11-01", 54.4],
      ["2021-11-02", 54.5],
      ["2021-11-03", 54.6],
      ["2021-11-04", 53.9],
      ["2021-11-05", 54.2],
      ["2021-11-06", 54.3],
      ["2021-11-07", 54.4],
      ["2021-11-08", 54.9],
      ["2021-11-09", 54.7],
      ["2021-11-10", 54.5],
      ["2021-11-11", 54.4],
      ["2021-11-12", 54.4],
      ["2021-11-13", 54.2],
      ["2021-11-14", 54.1],
      ["2021-11-15", 53.9],
             ],
             markArea: {
               itemStyle: {
                 color: 'rgba(255, 173, 177, 0.4)'
               },
               data: [
                 [
                   {
                     name: 'Latest Week',
                     xAxis: '2021-10-25'
                   },
                   {
                     xAxis: '2021-10-31'
                   }
                 ],
               ]
             }

    },
    {
      name: 'YR',
      type: 'line',
      smooth: 'true',
      data: [["2021-09-04", 58.9],
      ["2021-09-21", 55.4],
      ["2021-09-22", 55.1],
      ["2021-09-23", 55.3],
      ["2021-09-24", 55.4],
      ["2021-09-25", 55.6],
      ["2021-09-26", 55.3],
      ["2021-09-28", 55.0],
      ["2021-09-29", 54.7],
      ["2021-09-30", 54.5],
      ["2021-10-02", 54.4],
      ["2021-10-01", 54.4],
      ["2021-10-03", 54.2],
      ["2021-10-04", 54.3],
      ["2021-10-06", 54.6],
      ["2021-10-07", 54.9],
      ["2021-10-08", 54.7],
      ["2021-10-09", 54.6],
      ["2021-10-10", 54.2],
      ["2021-10-11", 54.4],
      ["2021-10-12", 54.2],
      ["2021-10-13", 54.4],
      ["2021-10-14", 54.4],
      ["2021-10-15", 53.8],
      ["2021-10-16", 53.8],
      ["2021-10-17", 54.0],
      ["2021-10-18", 53.8],
      ["2021-10-19", 53.8],
      ["2021-10-20", 54.0],
      ["2021-10-21", 54.0],
      ["2021-10-22", 54.1],
      ["2021-10-23", 53.8],
      ["2021-10-24", 53.5],
      ["2021-10-25", 53.1],
      ["2021-10-26", 53.0],
      ["2021-10-27", 52.9],
      ["2021-10-28", 52.8],
      ["2021-10-29", 53.1],
      ["2021-10-30", 52.9],
      ["2021-10-31", 53.0],
      ["2021-11-01", 53.0],
      ["2021-11-02", 52.7],
      ["2021-11-03", 52.7],
      ["2021-11-04", 53.4],
      ["2021-11-05", 53.3],
      ["2021-11-06", 53.3],
      ["2021-11-07", 53.2],
      ["2021-11-08", 53.4],
      ["2021-11-09", 53.4],
      ["2021-11-10", 53.5],
      ["2021-11-11", 53.3],
      ["2021-11-12", 53.3],
      ["2021-11-13", 53.1],
      ["2021-11-14", 53.0],
      ["2021-11-15", 53.0],
             ]
    }
  ]
};
{% endecharts %}


{% echarts 400 '85%' %}

option = {
  title: {
    text: 'Step Line'
  },
  tooltip: {
    trigger: 'axis'
  },
  legend: {
    data: ['Step Start', 'Step Middle', 'Step End']
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true
  },
  toolbox: {
    feature: {
      saveAsImage: {}
    }
  },
  xAxis: {
    type: 'time',
    boundaryGap: false
    },
  yAxis: [
    {type: 'value'},
      {
      name: 'Rainfall(mm)',
      nameLocation: 'start',
      max: 5,
      type: 'value',
      inverse: true
    }
    ],

  series: [
    {
      name: 'Step Start',
      type: 'line',
      step: 'start',
      data: [["2019-12-01", 1],
        ["2019-12-01", 14],
        ["2019-12-05", 83],
        ["2019-12-05", 84],
        ["2019-12-05", 100],
        ["2019-12-10", 262],
        ["2019-12-10", 263],
        ["2019-12-16", 281],
        ["2019-12-22", 637],
        ["2019-12-22", 643],
        ["2019-12-24", 724],
        ["2019-12-24", 725],
        ["2020-01-14", 829],
        ["2020-02-02", 2097],
        ["2020-02-03", 2177],
        ["2020-04-06", 2572],
        ["2020-04-08", 3024],
        ["2020-04-09", 3043],
        ["2020-04-09", 3198],
        ["2021-08-21", 4164],
        ["2021-10-10", 5475],
        ["2021-10-20", 6766],
        ["2021-10-23", 6885],
        ["2021-10-25", 6918],
        ["2021-10-25", 6933],
        ]
    },

  ]
};

{% endecharts %}







---

Google Chart test:


  <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
  <script type="text/javascript">
    google.charts.load('current', {'packages':['corechart']});
    google.charts.setOnLoadCallback(drawChart);

    function drawChart() {
      var data = google.visualization.arrayToDataTable([
        ['Year', 'Sales', 'Expenses'],
        ['2004',  1000,      400],
        ['2005',  1170,      460],
        ['2006',  660,       1120],
        ['2007',  1030,      540]
      ]);

      var options = {
        title: 'Company Performance',
        curveType: 'function',
        legend: { position: 'bottom' }
      };

      var chart = new google.visualization.LineChart(document.getElementById('curve_chart'));

      chart.draw(data, options);
    }
  </script>

<div id="curve_chart" style="width: 100%  ; height: 500px"></div>