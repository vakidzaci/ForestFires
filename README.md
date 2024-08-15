    filenames = os.listdir(WORK_DIR)
    filenames = [os.path.join(WORK_DIR, f) for f in filenames]
    # filenames.sort(key=lambda x: int(x[-3:]))
    read_tasks = group(read_page_task.s(path).set(queue='read_page_queue') for path in filenames)
    chord_result = chord(read_tasks)(collect_results.s().set(queue="collect_results_queue", task_id=task_id))
