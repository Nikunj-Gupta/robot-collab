import os 
import json
import openai
import re
import numpy as np
from rocobench.subtask_plan import LLMPathPlan
from typing import List, Tuple, Dict, Union, Optional, Any
from collections import defaultdict
from rocobench.envs import MujocoSimEnv
from transforms3d import euler, quaternions
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.envs.env_utils import Pose
from rocobench.envs import MujocoSimEnv, EnvState 

assert os.path.exists("openai_key.json"), "Please put your OpenAI API key in a string in robot-collab/openai_key.json"
OPENAI_KEY = str(json.load(open("openai_key.json")))
openai.api_key = OPENAI_KEY

# FEEDBACK_INSTRUCTION="""
# [Feedback Instruction]
# Follow these steps to improve [Environment Feedback]:
# 1) Decompose [Environment Feedback] into individual feedback for each robot.
# 2) If [Plan Passed] is False, identify each robot's contribution and penalize them individually.
# 3) If [Plan Passed] is True, identify each robot's contribution and reward them individually.
# """

# FEEDBACK_INSTRUCTION="""
# [Feedback Instruction]
# Follow these steps to improve [Environment Feedback]:
# 1) Decompose [Environment Feedback] into individual feedback for each robot.
# 2) Identify each robot's contribution to the task's success or failure and assign real-valued scores to them individually from 0 to 1. 0 is the lowest score and 1 is highest score.
# 3) Avoid attributing an equal score to each robot. In case of failure, penalize the final agent who came up with the plan.
# 4) Use any other computed values available from [Environment Feedback] to qualitatively determine each robot's scores. 
# """

# v1 (open-ended individual feedback)
v1_FEEDBACK_INSTRUCTION="""
[Feedback Instruction]
Given [Environment Feedback] follow these steps to improve feedback:
1) Clearly separate feedback for each robot.
2) Based on separated feedback, guide each robot to improve plan.
[How to Improve plan]
    If IK fails, propose more feasible step for the gripper to reach.
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects.
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)]
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""

# v2 (scoring-based individual feedback) 
v2_FEEDBACK_INSTRUCTION="""
[Feedback Instruction]
Given [Environment Feedback] follow these steps to improve feedback:
1) Clearly separate feedback for each robot.
2) Based on separated feedback, assign each robot a score of -1, 0 or 1 to improve plan. 0 if the robots action is WAIT. 
[How to use score]
   If -1, robot should choose more feasible steps in PATH, or choose different actions. 
   if 1, robot can maintain same (or similar) steps in PATH. 
[How to Improve plan]
    If IK fails, propose more feasible step for the gripper to reach.
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects.
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)]
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""

# v3 (reward-penalty-based individual feedback)
v3_FEEDBACK_INSTRUCTION="""
[Feedback Instruction]
Given [Environment Feedback] follow these steps to improve feedback:
1) Clearly separate feedback for each robot.
2) Based on separated feedback and computations from [Environment Feedback], reward or penalize each robot a numerical real value from -1.0 to 1.0 to improve plan.
[How to reward or penalize to improve plan]
   Give high penalty for causing significant failures. 
   Give low penalty for causing failures. 
   Give low reward for causing few failures. 
   Give high reward for successful plans or causing no failures. 
[How to use reward or penalty to improve plan]
   Aim is to maximize reward for each robot and improve plan.
   If high penalty, choose significantly more feasible steps in PATH, or choose significantly different actions that increase reward.
   If low penalty, choose more feasible steps in PATH, or choose different actions that increase reward.
   If low reward, choose slightly more feasible steps in PATH, or choose slightly different actions that increase reward.
   If high reward, robot can maintain same (or similar) steps in PATH and that maintain high reward.
[How to Improve plan]
    If IK fails, propose more feasible step for the gripper to reach.
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects.
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)]
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
Please strictly follow the following output format, example:
Individual Feedback for [Agent 1]:
    [Feedback for Agent 1]
Individual Feedback for [Agent 2]:
    [Feedback for Agent 2]"""

BINARY_FEEDBACK_INSTRUCTION="""
[Feedback Instruction]
Given [Environment Feedback] follow these steps to improve feedback:
1) Clearly separate feedback for each robot.
2) Based on separated feedback, provide individual rewards to each robot -1, 0 or +1.
[How to generate rewards]
    If robot's action is WAIT, generate 0 reward.
    If the robot fails to perform the stated action, provide a -1 reward based on the level of failure.
    If the robot succeeded in performing the action, provide a +1 reward.
"""

class FeedbackManager:
    """
    Takes in **parsed** LLM response, run task validations, and provide feedback if needed
    """
    def __init__(
        self,
        env: MujocoSimEnv,
        planner: MultiArmRRT,
        llm_output_mode: str = "action",
        robot_name_map: Dict[str, str] = {"panda": "Bob"}, 
        step_std_threshold: float = 0.1, # threshold for checking if the waypoint steps are evenly spaced
        max_failed_waypoints: int = 2,
        max_tokens: int = 512,
        temperature: float = 0,
        llm_source: str = "gpt-4",
        feedback_type: str = "textual"
    ):
        self.env = env
        self.planner = planner
        self.llm_output_mode = llm_output_mode
        self.robot_name_map = robot_name_map
        self.robot_agent_names = [v for k, v in robot_name_map.items()] 
        self.step_std_threshold = step_std_threshold
        self.max_failed_waypoints = max_failed_waypoints
        self.max_tokens = max_tokens
        self.feedback_type = feedback_type
        self.temperature = temperature
        self.llm_source = llm_source
        assert llm_source in ["gpt-4", "gpt-3.5-turbo", "claude", "gpt-4o-mini", "gpt-4o"], f"llm_source must be one of [gpt4, gpt-3.5-turbo, claude, gpt-4o-mini, gpt-4o], got {llm_source}"
    
    def get_full_path(self, llm_plan: LLMPathPlan) -> Dict[str, Pose]:
        full_path = dict()
        obs = self.env.get_obs()  
        for robot_name, agent_name in self.robot_name_map.items():
            robot_state = getattr(obs, robot_name)
            start_pose = robot_state.ee_pose.copy()
            target_pose = llm_plan.ee_targets[agent_name]
            if self.llm_output_mode == "action_and_path":
                full_path[agent_name] = [start_pose] + llm_plan.ee_waypoints[agent_name] + [target_pose]
            else:
                full_path[agent_name] = [start_pose, target_pose]
        return full_path        

    def task_feedback(self, llm_plan: LLMPathPlan) -> str: 
        task_feedback = self.env.get_task_feedback(
            llm_plan, 
            llm_plan.ee_target_poses
            ) 
        return task_feedback    

    def reach_feedback(self, pose_dict: Dict[str, Pose]) -> str:
        """ TODO: merge this into task env feedback """
        feedback = ""
        for agent_name, pose in pose_dict.items():
            if not self.env.check_reach_range(agent_name, pose.position):
                feedback += f"{agent_name} {pose.pos_string}; "
        return feedback

    def ik_feedback(self, pose_dict: Dict[str, Pose]) -> str:
        feedback = ""
        ik_result = self.planner.inverse_kinematics_all(self.env.physics, pose_dict)
        for name, result in ik_result.items():
            if result is None:
                pose = pose_dict[name]
                feedback += f"{name} {pose.pos_string}; "
        return feedback, ik_result
    
    def collision_feedback(
        self, 
        llm_plan: LLMPathPlan, 
        ik_result: Dict[str, np.ndarray]
    ) -> str:
        assert all([result is not None for result in ik_result.values()]), "Collision feedback should be called after ik feedback"
        feedback = ""
        target_qpos = np.concatenate(
            [ik_result[name][0] for name in self.robot_agent_names]
            ) 
        # inhand_ids = llm_plan.get_inhand_ids(self.env.physics)
        allowed_collision_ids = llm_plan.get_allowed_collision_ids(self.env.physics)
        self.planner.set_inhand_info(
            self.env.physics,
            llm_plan.get_inhand_obj_info(self.env.physics)
            )
        collided_body_pairs = self.planner.get_collided_links(
            qpos=target_qpos, 
            physics=self.env.physics,
            allow_grasp=True, # check ids should already be set in at policy __init__
            check_grasp_ids=allowed_collision_ids, 
            show=0,
            )

        adjusted_names = []
        for name1, name2 in collided_body_pairs:
            name1 = self.robot_name_map.get(name1, name1) # convert panda to Bob 
            name2 = self.robot_name_map.get(name2, name2)
            adjusted_names.append((name1, name2))

        if len(collided_body_pairs) > 0: 
            # make a string in the format [name: (x,y,z), name: (x,y,z), ...]
            feedback = "collided object pairs: "
            feedback += ", ".join(
                [f"{name1}-{name2}" for name1, name2 in adjusted_names]
                )
        return feedback

    def path_feedback(self, llm_plan: LLMPathPlan) -> str:
        """ check if the waypoint steps are evenly spaced """
        feedback = ""
        if self.llm_output_mode == "action_and_path":
            full_path = self.get_full_path(llm_plan)
            for agent_name, path in full_path.items():
                stepwise_dist = []
                step_pairs = []
                for i in range(len(path)-1):
                    stepwise_dist.append(np.linalg.norm(path[i+1][:3] - path[i][:3]))
                    x,y,z = path[i][:3]
                    x2,y2,z2 = path[i+1][:3]
                    step_pairs.append(
                        f"({x:.2f},{y:.2f},{z:.2f})-({x2:.2f},{y2:.2f},{z2:.2f})"
                        )
                stepwise_dist = np.array(stepwise_dist)
                max_dist_pair = f"  Distance between {step_pairs[np.argmax(stepwise_dist)]} is {np.max(stepwise_dist):.2f}, too high"
                min_dist_pair = f"  Distance between {step_pairs[np.argmin(stepwise_dist)]} is {np.min(stepwise_dist):.2f}, too low"
                _std = np.std(stepwise_dist)
                if _std > self.step_std_threshold: 
                    feedback += f"You must make {agent_name}'s path more evenly spaced:\n{max_dist_pair}\n{min_dist_pair}\n  Overall Distance std: {_std:.2f}" 
        return feedback
    
    def get_step_string(self, pose_dict) -> str:
        step_string = ""
        for agent_name, pose in pose_dict.items():
            step_string += f"{agent_name} {pose.pos_string}; "
        return step_string[:-2]

    def single_step_feedback(self, llm_plan, pose_dict, step_type: str = "Goal") -> Tuple[bool, str]:
        step_string = self.get_step_string(pose_dict)
        feedback = f"{step_type} Step {step_string}:\n  "
        reach = self.reach_feedback(pose_dict)
        all_passed = True
        if len(reach) > 0:
            all_passed = False
            feedback += f" - Reachability failed: Out of reach: {reach}\n  "
        else:
            # feedback += " - Reach feedback: passed\n  "
            ik_feedback, ik_result = self.ik_feedback(pose_dict)
            if len(ik_feedback) > 0:
                all_passed = False
                feedback += f" - IK failed: on {ik_feedback}\n  "
            else:
                # feedback += " - IK feedback: passed\n  "
                collision_feedback = self.collision_feedback(llm_plan, ik_result)
                if len(collision_feedback) > 0:
                    all_passed = False
                    feedback += f" - Collision detected: {collision_feedback}\n  "
                # else:
                #     feedback += " - Collision feedback: passed\n  "
        if all_passed:
            # feedback = f"{step_type} Step {step_string}: All checks passed\n"
            feedback = ""
        return all_passed, feedback

    def compose_system_prompt(
        self, 
        plan_passed,
        feedback
    ) -> str:
        feedback_desc = f"{feedback}\n [Plan Passed] is {plan_passed}\n" 
        if self.feedback_type == "v1":
            feedback_desc += v1_FEEDBACK_INSTRUCTION
        elif self.feedback_type == "v2":
            feedback_desc += v2_FEEDBACK_INSTRUCTION
        elif self.feedback_type == "v3":
            feedback_desc += v3_FEEDBACK_INSTRUCTION
        elif self.feedback_type == "binary":
            feedback_desc += BINARY_FEEDBACK_INSTRUCTION
        system_prompt = f"{feedback_desc}\n" 
        return system_prompt 
 
    def query_once(self, system_prompt, max_query=3):
        response = None
        usage = None   
        # print('======= system prompt ======= \n ', system_prompt)
        for n in range(max_query):
            print('querying {}th time'.format(n))
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm_source, 
                    messages=[
                        # {"role": "user", "content": ""},
                        {"role": "system", "content": system_prompt}, 
                    ],
                    max_tokens=self.max_tokens, 
                    temperature=self.temperature,
                    )
                usage = response['usage']
                response = response['choices'][0]['message']["content"]
                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except:
                print("API error, try again")
            continue
        # breakpoint()
        return response, usage
    
    def give_feedback(self, llm_plan: LLMPathPlan) -> Tuple[bool, str]:
        """
        Given a parsed LLM plan, run task validations and provide feedback if needed
        """
        feedback = f"[Environment Feedback]:\n- Previous Plan:\n{llm_plan.parsed_proposal}\n"
        task_feedback = self.task_feedback(llm_plan)
        plan_passed = True 
        if len(task_feedback) == 0:
            # feedback += "- Task Constraints: all satisfied\n"
            target_passed, target_feedback = self.single_step_feedback(
                llm_plan, llm_plan.ee_target_poses, "- Goal")
            # if not target_passed:
            #     print(target_feedback)
            #     breakpoint()
            feedback += target_feedback
            plan_passed = plan_passed and target_passed

            if self.llm_output_mode == "action_and_path":
                failed_waypoints = 0 
                for pose_dict in llm_plan.ee_waypoints_list:
                    step_passed, step_feedback = self.single_step_feedback(llm_plan, pose_dict, "- Waypoint")
                    feedback += step_feedback
                    if not step_passed:
                        failed_waypoints += 1 

                waypoints_passed = failed_waypoints <= self.max_failed_waypoints  
                plan_passed = plan_passed and waypoints_passed
                if waypoints_passed:              
                    # feedback += f"All waypoint steps: passed\n"
                    path_feedback = self.path_feedback(llm_plan)
                    if len(path_feedback) > 0:
                        feedback += f"- Path feedback: failed, {path_feedback}\n"
                        plan_passed = False 
            
        else:
            plan_passed = False
            feedback += f"Task Constraints:\n failed, {task_feedback}\n"
        # breakpoint()
        
        feedback_dict = None
        if self.feedback_type in ["v1", "v2", "v3"]:
            system_prompt = self.compose_system_prompt(plan_passed, feedback) 
            response, usage = self.query_once(
                system_prompt, 
                max_query=3,
            )

        # # append Individual Feedback 
        # feedback += f"[Individual Feedback]:\n{response}\n"

        # Overwrite [Environment Feedback] with [Individual Feedback]
        #feedback_dict = None
        #if self.feedback_type in ["v1", "v2", "v3"]:
            feedback_dict = self.parse_feedback(response, llm_plan.parsed_proposal)
            # TODO add f"[Environment Feedback]:\n- Previous Plan:\n{llm_plan.parsed_proposal}\n{respons}\n" to each value of feedback dict

            feedback = f"[Environment Feedback]:\n- Previous Plan:\n{llm_plan.parsed_proposal}\n{response}\n"
        
        #feedback_dict = None
        #if self.feedback_type in ["v1", "v2", "v3"]:
        #    feedback_dict = self.parse_feedback(feedback)
        #breakpoint()

        return plan_passed, feedback, feedback_dict

    def parse_feedback(self, feedback: str=None, parsed_plan: str=None):
        """
            For each generated feedback, parse individualized feedback and return feedback per agent. If there is no individualized feedback, retry
        """
        agent_names = list(self.robot_name_map.values())
        feedback_dict = dict()

        feedback_split = feedback.strip().split("Individual Feedback for ")
        feedback_split = [f.strip() for f in feedback_split if len(f.strip()) != 0]
        feedback_split = [[f.split("\n")[0].strip(": "), "\n".join(f.split("\n")[1:]).strip()] for f in feedback_split]
        feedback_dict = {agent_name: f"[Environment Feedback]:\n- Previous Plan:\n{parsed_proposal}\n{feedback_agent}" for agent_name, feedback_agent in feedback_split if agent_name in agent_names}
        print(f"Robot Names: {agent_names}")
        print(f"Feedback : {feedback}")
        print(f"Feedback Split:\n{feedback_split}")
        assert len(feedback_dict) == len(agent_names)

        return feedback_dict

