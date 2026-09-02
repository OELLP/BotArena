<script setup>
defineProps({ title: String, result: Object })

function riskText(result) {
  return result ? `${(result.risk_score * 100).toFixed(1)}% · ${result.risk_level}` : '等待检测'
}

const labels = { behavior_agent: '行为智能体', text_agent: '文本智能体', relation_agent: '关系智能体', propagation_agent: '传播智能体' }
</script>

<template>
  <article class="panel result">
    <span class="eyebrow">{{ title }} · {{ result?.model_version || '等待模型' }}</span>
    <strong :class="result?.risk_level">{{ riskText(result) }}</strong>
    <div v-if="result" class="scores">
      <div v-for="(score, name) in result.agent_scores" :key="name">
        <span>{{ labels[name] || name }} · {{ result.agent_models?.[name] }}</span><b>{{ (score * 100).toFixed(1) }}%</b>
        <i><em :style="{ width: `${score * 100}%` }" /></i>
        <small>置信度 {{ ((result.agent_confidence?.[name] || 0) * 100).toFixed(0) }}% · 数据覆盖 {{ ((result.data_coverage?.[name] || 0) * 100).toFixed(0) }}%</small>
      </div>
    </div>
    <ul v-if="result"><li v-for="item in result.evidence" :key="item">{{ item }}</li></ul>
  </article>
</template>
