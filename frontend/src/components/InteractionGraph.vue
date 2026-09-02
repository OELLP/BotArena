<script setup>
import * as echarts from 'echarts/core'
import { GraphChart } from 'echarts/charts'
import { TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

echarts.use([GraphChart, TooltipComponent, CanvasRenderer])

const props = defineProps({ graph: { type: Object, required: true } })
const element = ref()
const mode = ref('all')
const selected = ref(null)
let chart
let observer

const modes = [
  { id: 'all', label: '微博全景' },
  { id: 'publish', label: '发布关系' },
]
const nodeLegend = [
  { type: 'account', label: '目标微博账号', color: '#7c3aed', shape: 'diamond' },
  { type: 'post', label: '微博内容', color: '#f97373', shape: 'square' },
  { type: 'comment', label: '评论用户', color: '#67d8c8' },
  { type: 'repost', label: '转发用户', color: '#60a5fa' },
  { type: 'attitude', label: '点赞用户', color: '#a3e635' },
  { type: 'multi', label: '多行为用户', color: '#facc15' },
]
const edgeLegend = [
  { action: 'publish', label: '发布微博', color: '#a855f7' },
  { action: 'comment', label: '评论微博', color: '#2dd4bf' },
  { action: 'repost', label: '转发微博', color: '#3b82f6' },
  { action: 'attitude', label: '点赞微博', color: '#84cc16' },
]
const typeIndex = Object.fromEntries(nodeLegend.map((item, index) => [item.type, index]))
const typeLabel = Object.fromEntries(nodeLegend.map(item => [item.type, item.label]))
const actionLabel = Object.fromEntries(edgeLegend.map(item => [item.action, item.label]))
const edgeColor = Object.fromEntries(edgeLegend.map(item => [item.action, item.color]))

function render() {
  if (!chart) return
  const allowed = mode.value === 'publish'
    ? new Set(['publish'])
    : null
  const sourceLinks = (props.graph?.links ?? []).filter(link => !allowed || allowed.has(link.action))
  const visibleIds = new Set(sourceLinks.flatMap(link => [link.source, link.target]))
  const nodes = (props.graph?.nodes ?? [])
    .filter(node => !allowed || visibleIds.has(node.id))
    .map(node => ({
      ...node,
      category: typeIndex[node.type] ?? 2,
      symbol: node.type === 'account' ? 'diamond' : node.type === 'post' ? 'rect' : 'circle',
      symbolSize: node.type === 'account' ? 54 : node.type === 'post' ? [42, 32] : Math.min(34, 16 + (node.degree ?? 0) * 3),
      label: { show: node.type === 'account' || node.type === 'post' },
      emphasis: { label: { show: true } },
    }))
  const links = sourceLinks.map(link => ({
    ...link,
    lineStyle: { color: edgeColor[link.action] ?? '#94a3b8' },
  }))

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      confine: true,
      formatter(params) {
        if (params.dataType === 'edge') return escapeHtml(actionLabel[params.data.action] ?? '互动关系')
        const item = params.data
        if (item.type === 'post') {
          return `<b>${escapeHtml(item.name)}</b><br/>${escapeHtml(shorten(item.content, 90))}<br/>转发 ${item.repost_count} · 评论 ${item.comment_count} · 点赞 ${item.attitude_count}`
        }
        return `<b>${escapeHtml(item.name)}</b><br/>${escapeHtml(typeLabel[item.type] ?? '互动用户')} · ${item.degree ?? 0} 个连接`
      },
    },
    series: [{
      type: 'graph',
      layout: 'force',
      roam: true,
      draggable: true,
      data: nodes,
      links,
      categories: nodeLegend.map(item => ({ name: item.label, itemStyle: { color: item.color } })),
      label: { show: false, color: '#172033', fontSize: 11, position: 'right' },
      edgeSymbol: ['none', 'arrow'],
      edgeSymbolSize: 6,
      lineStyle: { width: 1.4, opacity: 0.64, curveness: 0.06 },
      emphasis: { focus: 'adjacency', lineStyle: { width: 3, opacity: 1 } },
      force: { repulsion: 155, edgeLength: [62, 115], gravity: 0.07 },
    }],
  }, true)
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char])
}

function shorten(value, limit) {
  const text = String(value ?? '')
  return text.length > limit ? `${text.slice(0, limit)}…` : text
}

onMounted(() => {
  chart = echarts.init(element.value)
  chart.on('click', params => { if (params.dataType === 'node') selected.value = params.data })
  observer = new ResizeObserver(() => chart.resize())
  observer.observe(element.value)
  render()
})
watch([() => props.graph, mode], () => { selected.value = null; render() }, { deep: true })
onBeforeUnmount(() => { observer?.disconnect(); chart?.dispose() })
</script>

<template>
  <article class="panel graph-panel">
    <div class="graph-heading">
      <div><span class="eyebrow">微博模块</span><h2>微博传播行为表征图</h2></div>
      <div class="graph-summary">
        <span>{{ graph.summary.post_count }} 条微博</span>
        <span>{{ graph.summary.user_count }} 个互动用户</span>
        <span>{{ graph.summary.edge_count }} 条关系</span>
        <span>关系覆盖率 {{ Math.round(graph.summary.coverage * 100) }}%</span>
      </div>
    </div>

    <div class="graph-workspace">
      <div ref="element" class="interaction-chart" role="img" aria-label="微博传播行为表征图" />
      <div class="graph-side">
        <div class="graph-modes" aria-label="关系图模块">
          <button v-for="item in modes" :key="item.id" type="button" :class="{ active: mode === item.id }" @click="mode = item.id">{{ item.label }}</button>
        </div>

        <h3>节点表征</h3>
        <div class="legend-list">
          <div v-for="item in nodeLegend" :key="item.type"><i class="node-mark" :class="item.shape" :style="{ background: item.color }" /><span>{{ item.label }}</span></div>
        </div>
        <h3>传播关系</h3>
        <div class="legend-list edge-list">
          <div v-for="item in edgeLegend" :key="item.action"><i :style="{ background: item.color }" /><span>{{ item.label }}</span></div>
        </div>

        <div class="node-detail">
          <template v-if="selected">
            <small>节点详情</small><strong>{{ selected.name }}</strong>
            <p>{{ typeLabel[selected.type] }} · {{ selected.degree }} 个连接</p>
            <p v-if="selected.type === 'post'">{{ shorten(selected.content, 110) }}</p>
            <p v-if="selected.type === 'post'">转发 {{ selected.repost_count }} · 评论 {{ selected.comment_count }} · 点赞 {{ selected.attitude_count }}</p>
            <p v-if="selected.actions?.length">参与行为：{{ selected.actions.map(item => actionLabel[item]).join('、') }}</p>
          </template>
          <p v-else>点击图中的账号、微博或用户节点，可查看对应表征信息。</p>
        </div>
      </div>
    </div>

    <p class="graph-note">图中使用最近 {{ graph.summary.post_count }} 条可访问微博构建账号—内容—用户网络。最近一条微博每类最多采样 {{ graph.summary.sample_limit }} 个用户，其余微博每类最多 {{ graph.summary.secondary_sample_limit }} 个；相应接口受限时，该类节点可能为 0。</p>
  </article>
</template>
