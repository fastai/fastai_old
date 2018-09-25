{%- extends 'hide.tpl' -%}{% block body %}---
title: {{resources.title}}
keywords: {{resources.keywords}}
sidebar: home_sidebar
tags: {{resources.tags}}
summary: {{resources.summary}}
---

<div class="container" id="notebook-container">
    {{ super()  }}
</div>
{%- endblock body %}
