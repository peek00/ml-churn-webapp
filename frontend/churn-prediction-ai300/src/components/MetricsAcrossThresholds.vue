<script setup>
import Plotly from 'plotly.js-dist';
import { ref, onMounted } from 'vue';
import metrics_across_thresholds from '../assets/metrics_across_thresholds.json';

const plot = ref("plot");
onMounted(() => {
  const keys = Object.keys(metrics_across_thresholds);
  const accuracyValues = Object.values(metrics_across_thresholds).map(item => item.accuracy);
  const tooltips = keys.map(key => {
    const metrics = metrics_across_thresholds[key];
    return `Key: ${key}<br>Accuracy: ${metrics.accuracy}<br>Precision: ${metrics.precision}<br>Recall: ${metrics.recall}<br>F1: ${metrics.f1}<br>Factor: ${metrics.factor}`;
  });
  // Plotting
  const data = {
    x: keys,
    y: accuracyValues,
    hovertemplate: tooltips.map(tooltip => `<b>${tooltip}</b><extra></extra>`),
    type: 'scatter',
    mode: 'lines+markers',
  };

  const layout = {
    xaxis: {
      title: 'Thresholds'
    },
    yaxis: {
      title: 'Accuracy'
    },
    hovermode: 'closest',
    title:{
      text: `Metrics Across Thresholds`,
        font: { size: 20 },
        automargin: true,
        xanchor: 'center',
        x: 0.5,
        pad: { t: 20 },
    }
  };

  Plotly.newPlot('plot', [data], layout);
});
</script>

<template>
  <div id="plot" :ref="plot" style="width: 1000px; height: 500px;"> </div>
</template>