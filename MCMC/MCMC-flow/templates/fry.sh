{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --job-name="{{ id }}"
{% if partition %}
#SBATCH --partition={{ partition }}
{% endif %}
{% if walltime %}
#SBATCH -t {{ 48|format_timedelta }}
{% endif %}
{% if job_output %}
#SBATCH --output={{ job_output }}
#SBATCH --error={{ job_output }}
{% endif %}
{% block tasks %}
#SBATCH --ntasks={{ np_global }}
{% endblock %}
{% endblock %}