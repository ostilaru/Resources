SUBDIRS = src/sac2spec_c1 src/sac2spec_c9 src/xc_mono src/xc_dual src/stack src/rotate

.PHONY: $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	@echo "Making in directory $@"
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		echo "Cleaning in directory $$dir"; \
		$(MAKE) -C $$dir clean; \
	done

veryclean:
	for dir in $(SUBDIRS); do \
		echo "Performing very clean in directory $$dir"; \
		$(MAKE) -C $$dir veryclean; \
	done
