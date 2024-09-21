### removewhite.py
### julia margie
### clean up data for graphs

newdoc = ""
with open("with_facts.tsv", 'r') as doc:
	lines = doc.readlines()

	for index, line in enumerate(lines):
		if not line[0:11].isupper():
			lines[index - 1]=lines[index-1].strip()
		if line == '\n':
			line = ""
	newdoc = "".join(lines)	

with open("test_f.tsv", 'w') as doc:
	doc.write(newdoc)
