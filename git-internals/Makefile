.PHONY: .slides.html

slides.html:
	pandoc \
		--filter pandoc-include-code \
		-t revealjs \
		-s presentation/slides.md \
		-o .slides.html \
		--highlight-style zenburn \
		-V revealjs-url=https://unpkg.com/reveal.js@^4/ \
		-V theme=black \
		--css black-root.css \
