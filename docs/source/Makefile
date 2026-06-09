# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXBUILD        = sphinx-build
SPHINXOPTS         = -j auto
SPHINXAUTOBUILD    = sphinx-autobuild
SPHINXAUTOOPTS     = -j auto
SOURCEDIR          = .
BUILDDIR           = ../build
DIR          = ../../src/llm4series

.PHONY: help build autobuild

# "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf $(BUILDDIR)
	@echo "Deleted directory $(BUILDDIR)."

# $(O) is meant as a shortcut for custom options.
# i.e to log stderr into a separate file:
# make build O="--no-color 2> build_warnings.log"
html:
	./symlink_examples.sh
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# $(O) is meant as a shortcut for custom options.
autobuild:
	$(SPHINXAUTOBUILD) \
	"$(SOURCEDIR)" "$(BUILDDIR)/html" \
	-d "$(BUILDDIR)/doctrees" \
	$(SPHINXAUTOOPTS) $(O) \
	--watch "$(DIR)" \
	--re-ignore ".*\.json" \
  --pre-build "./symlink_examples.sh"
