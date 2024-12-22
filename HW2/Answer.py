# AmirReza Azari
# 99101087

import collections
import json


def main():
    with open('input.json', 'r') as file:
        raw_data = json.load(file)

    children_per_parent = collections.defaultdict(list)
    parent_for_child = {}
    task_info = {}

    for task in raw_data:
        task_id = task["taskId"]
        parent_id = task["parentId"]
        task_info[task_id] = task

        if parent_id is None:
            children_per_parent[task_id] = []
        else:
            children_per_parent[parent_id].append(task_id)
            parent_for_child[task_id] = parent_id

    in_degree_map = {task_id: 0 for task_id in task_info.keys()}

    for children in children_per_parent.values():
        for child in children:
            in_degree_map[child] += 1

    ready_tasks_queue = collections.deque([task_id for task_id, degree in in_degree_map.items() if degree == 0])
    sorted_task_ids = []

    while ready_tasks_queue:
        current_task_id = ready_tasks_queue.popleft()
        sorted_task_ids.append(current_task_id)
        for child_id in children_per_parent[current_task_id]:
            in_degree_map[child_id] -= 1
            if in_degree_map[child_id] == 0:
                ready_tasks_queue.append(child_id)

    for task_id in sorted_task_ids:
        task = task_info[task_id]
        parent_task_id = parent_for_child.get(task_id)
        if parent_task_id is not None:
            parent_task = task_info[parent_task_id]
            task["releasedTime"] = max(
                task["releasedTime"],
                parent_task["releasedTime"] + parent_task["executionTime"]
            )

    for task_id in reversed(sorted_task_ids):
        children = children_per_parent[task_id]
        task = task_info[task_id]
        updated_deadlines = [task["deadline"]]
        for child_id in children:
            child_task = task_info[child_id]
            updated_deadlines.append(child_task['deadline'] - child_task['executionTime'])
        task["deadline"] = min(updated_deadlines)

    tasks_to_schedule = list(task_info.values())

    for task in tasks_to_schedule:
        task['remainingExecutionTime'] = task['executionTime']

    completed_task_map = {}
    final_scheduled_tasks = []
    current_time = 0
    active_task = None
    is_schedulable = True

    while tasks_to_schedule or active_task:
        available_tasks = [
            task for task in tasks_to_schedule
            if task['releasedTime'] <= current_time
               and (task['parentId'] is None or task['parentId'] in completed_task_map)
        ]

        if active_task:
            if active_task['remainingExecutionTime'] - 1 >= 0:
                available_tasks.append(active_task)

        def custom_sort(task):
            return task['deadline'], task['taskId']

        def task_id_sort(task):
            return task["taskId"]

        available_tasks.sort(key=custom_sort)

        if available_tasks:
            selected_task = available_tasks[0]
        else:
            if tasks_to_schedule:
                next_release_time = min(task['releasedTime'] for task in tasks_to_schedule)
                current_time = next_release_time
                continue
            else:
                break

        selected_task = available_tasks[0]

        if active_task and active_task['taskId'] != selected_task['taskId']:
            active_task['releasedTime'] = current_time
            tasks_to_schedule.append(active_task)

        if selected_task in tasks_to_schedule:
            tasks_to_schedule.remove(selected_task)

        start_time = current_time
        end_time = current_time + 1
        selected_task['startTime'] = start_time

        if end_time > selected_task['deadline']:
            is_schedulable = False

        if 'executionTimes' not in selected_task:
            selected_task['executionTimes'] = []

        selected_task['executionTimes'].append([start_time, end_time])
        current_time = end_time
        selected_task['remainingExecutionTime'] -= 1

        if selected_task['remainingExecutionTime'] <= 0:
            completed_task_map[selected_task['taskId']] = {
                "taskId": selected_task['taskId'], "releasedTime": selected_task['releasedTime'],
                "deadline": selected_task['deadline'], "executionTimes": selected_task['executionTimes'],
                "parentId": selected_task.get('parentId'),
            }
            final_scheduled_tasks.append(completed_task_map[selected_task['taskId']])
            active_task = None
        else:
            active_task = selected_task

    for task in final_scheduled_tasks:
        merged_intervals = []
        current_interval = task['executionTimes'][0]

        for i in range(1, len(task['executionTimes'])):
            next_interval = task['executionTimes'][i]
            if current_interval[1] == next_interval[0]:
                current_interval[1] = next_interval[1]
            else:
                merged_intervals.append(current_interval)
                current_interval = next_interval

        merged_intervals.append(current_interval)
        task['executionTimes'] = merged_intervals

    final_scheduled_tasks.sort(key=task_id_sort)

    output_tasks = [
        dict(taskId=task["taskId"], releasedTime=task["releasedTime"], deadline=task["deadline"],
             executionTimes=task["executionTimes"])
        for task in final_scheduled_tasks
    ]

    with open("result.json", "w") as output_file:
        json.dump(output_tasks, output_file)

    if not is_schedulable:
        print("Not Schedulable")
    else:
        print("Schedulable")


if __name__ == '__main__':
    main()
