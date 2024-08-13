def monitor_directory():
        processed_files = set()
        while process.poll() is None or process.returncode == 0:  # Continue until the process is done
            current_files = set(os.listdir(output_dir))
            new_files = current_files - processed_files
            for filename in sorted(new_files):
                if filename.endswith(".png"):
                    page_path = os.path.join(output_dir, filename)
                    # Enqueue read_page_task for each new page and add it to read_tasks list
                    task = read_page_task.s(page_path).set(queue='read_page_queue')
                    read_tasks.append(task)
                    processed_files.add(filename)
            time.sleep(0.5)
