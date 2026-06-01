---
layout: page
permalink: /publications/
title: publications
description: This page may not be up to date. For the most up to date information, please visit my <a href="https://scholar.google.com/citations?user=ml44TKkAAAAJ&hl=en">google scholar page</a>.
years: [2024, 2023, 2022, 2021, 2019, 2018]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
