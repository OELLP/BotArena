<script setup>
import * as echarts from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TitleComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

echarts.use([BarChart, GridComponent, TitleComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({ title: String, rows: { type: Array, default: () => [] } })
const element = ref()
let chart
let observer

function render() {
  if (!chart) return
  chart.setOption({
    backgroundColor: 'transparent',
    title: { text: props.title, textStyle: { color: '#172033', fontSize: 15 } },
    tooltip: { trigger: 'axis' },
    grid: { left: 46, right: 18, top: 52, bottom: 34 },
    xAxis: { type: 'category', data: props.rows.map(row => row.name), axisLabel: { color: '#64748b' }, axisLine: { lineStyle: { color: '#cbd5e1' } } },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: '#e5eaf2' } } },
    series: [{ type: 'bar', data: props.rows.map(row => row.value), barWidth: 28, itemStyle: { color: '#2563eb', borderRadius: [6, 6, 0, 0] } }],
  })
}

onMounted(() => {
  chart = echarts.init(element.value)
  observer = new ResizeObserver(() => chart.resize())
  observer.observe(element.value)
  render()
})
watch(() => props.rows, render, { deep: true })
onBeforeUnmount(() => { observer?.disconnect(); chart?.dispose() })
</script>

<template><div ref="element" class="chart" role="img" :aria-label="title" /></template>
