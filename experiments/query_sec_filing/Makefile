# Makefile that provide some convenience commands

# Folders that where created during the execution of the code
CLEAN_DIRS = embedding_model filings_index filings

.PHONY: clean

clean:
    # This command will clean the predefined folders
    @echo "Cleaning up folders..."
    @for dir in $(CLEAN_DIRS); do \
        if [ -d "$$dir" ]; then \
            rm -rf "$$dir"/*; \
            echo "Cleaned $$dir"; \
        else \
            echo "$$dir does not exist"; \
        fi \
    done
