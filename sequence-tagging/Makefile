.PHONY: .slides.html sim

# Define variables
SRC = slides.md
DST = .slides.html
CSS = static/custom.css
PLUGIN_CONFIG = static/reveal-plugin-config.html

reveal_flags = --mathjax \
		--slide-level 3 \
		-V revealjs-url=https://unpkg.com/reveal.js@^4/ \
		-V theme=black \
		-V transition="linear" \
		-V slideNumber="'c/t'" \
		--css static/custom.css \
		--css static/custom-elements.css \
		--bibliography ./bibliography.bib \
		--citeproc \
		--no-highlight \
		--filter pandoc-codeblock-include


slides.html:
	pandoc\
		-t revealjs \
	  	-s $(SRC) \
		-o $(DST) \
		--template static/pandoc-template.revealjs \
		$(reveal_flags)
