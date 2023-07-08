<script setup>
import Plotly from 'plotly.js-dist';
import { ref, onMounted } from 'vue';
import roc_curve from '../assets/roc_curve.json';

const plot = ref("plot");
onMounted(() => {
  const fpr = roc_curve.fpr;
  const tpr = roc_curve.tpr;
  const thresholds = roc_curve.thresholds;

  const fig = {
    data: [
      {
        x: fpr,
        y: tpr,
        type: 'scatter',
        fill: 'tozeroy',
        mode: 'lines',
        line: { shape: 'hv' },
      },
    ],
    layout: {
      xaxis: {
        title: 'False Positive Rate',
      },
      yaxis: {
        title: 'True Positive Rate',
        scaleanchor: 'x',
        scaleratio: 1,
      },
      title: {
        text: `ROC Curve`,
        font: { size: 20 },
        automargin: true,
        xanchor: 'center',
        x: 0.5,
        pad: { t: 20 },
      },
    },
  };

  if (thresholds) {
    const hovertemplate =
      'Threshold: %{text}<br>FPR: %{x}<br>TPR: %{y}<extra></extra>';
    fig.data[0].text = thresholds.map((threshold) => threshold.toFixed(4));
    fig.data[0].hovertemplate = hovertemplate;
  }
  Plotly.newPlot('plot2', fig.data, fig.layout);



});
</script>

<template>
  <div id="plot2" :ref="plot" style="width: 1000px; height: 500px;"> </div>
</template>