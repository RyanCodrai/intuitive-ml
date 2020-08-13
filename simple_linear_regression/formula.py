import re
from urllib.parse import quote

with open("Simple Linear Regression.md", "r") as f:
	content = f.read()


formula = re.compile(r"(?:```python.*?```)|(\$\$?.+?\$\$?)", re.MULTILINE | re.DOTALL)

offset = 0
for m in formula.finditer(content):
	if m.group(1) is None:
		continue

	equation = m.group(1)

	if equation[:2] == "$$":
		# Remove the extra dollar sign at the start and dn
		equation = equation[1:-1]
		# Set the template to correspond to a centered formula
		equation_template = "<div align='center'><img src='https://render.githubusercontent.com/render/math?math=%s'></div>"
	else:
		equation_template = "<img valign='middle' src='https://render.githubusercontent.com/render/math?math=%s'>"
	
	github_formula = equation_template % quote(equation)
	content = content[:m.start(1) + offset] + github_formula + content[m.end(1) + offset:]
	offset += len(github_formula) - len(m.group(1))

with open("README.md", 'w') as f:
	f.write(content)