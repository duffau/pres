.PHONY: .slides.html sim

# Define variables
SRC = slides.md
DST = .slides.html
CSS = static/custom.css
PLUGIN_CONFIG = static/reveal-plugin-config.html

reveal_flags = --mathjax \
		--highlight-style zenburn \
		--slide-level 3 \
		-V revealjs-url=https://unpkg.com/reveal.js@^4/ \
		-V theme=black \
		-V transition="linear" \
		-V slideNumber="'c/t'" \
		--css static/custom.css \
		--css static/custom-elements.css \
		--bibliography ./bibliography.bib \
		--citeproc

slides.html:
	pandoc -t revealjs \
	  	-s $(SRC) \
		-o $(DST) \
		$(reveal_flags)
	# @echo "Replacing reveal.js plugins section in $(DST)"
	# @sed '/<!-- reveal.js plugins -->/q' $(DST) > tmp && \
	# cat $(PLUGIN_CONFIG) >> tmp && \
	# mv tmp $(DST)