{%- extends 'basic.tpl' -%}{% block body %}---
title: {{resources.title}}
keywords: {{resources.keywords}}
tags: {{resources.tags}}
sidebar: home_sidebar
summary: {{resources.summary}}
---

    <div class="container" id="notebook-container">
{{ super() }}
    </div>
{%- endblock body %}

