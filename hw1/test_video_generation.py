#!/usr/bin/env python3
"""
Test script to verify video generation and environment visualization for hw1.
"""

import os
import sys
import gym
import numpy as np
import torch
from pyvirtualdisplay import Display

# Add the project directory to the path
sys.path.append('/home/shuai/Projects/16831-S25/hw1')

from rob831.infrastructure import utils
from rob831.policies.loaded_gaussian_policy import LoadedGaussianPolicy

# Try to import gnwrapper for video testing (like in ipynb)
try:
    import gnwrapper
    HAS_GNWRAPPER = True
except ImportError:
    HAS_GNWRAPPER = False
    print("⚠️ gnwrapper not available, will use alternative video testing")

def test_environment_rendering():
    """Test if the Ant environment can be rendered properly (following ipynb approach)."""
    print("Testing Ant environment rendering...")
    
    try:
        # Start virtual display for headless server
        display = Display(visible=0, size=(1400, 900))
        display.start()
        print("✅ Virtual display started")
        
        if HAS_GNWRAPPER:
            # Use gnwrapper like in ipynb
            print("Testing with gnwrapper (like ipynb)...")
            env = gnwrapper.LoopAnimation(gym.make('Ant-v2'))
            observation = env.reset()
            print(f"✅ Ant-v2 environment created with gnwrapper, observation shape: {observation.shape}")
            
            # Run a few steps like in ipynb
            for i in range(20):
                obs, rew, term, _ = env.step(env.action_space.sample())
                env.render()
                if term:
                    break
            
            print("✅ Environment simulation with gnwrapper working")
            env.close()
        else:
            # Fallback to regular gym
            print("Testing with regular gym...")
            env = gym.make('Ant-v2')
            obs = env.reset()
            print(f"✅ Ant-v2 environment created, observation shape: {obs.shape}")
            
            # Test a few steps with random actions
            for i in range(10):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
            
            print("✅ Environment simulation working")
            env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Environment rendering failed: {e}")
        return False

def test_expert_policy():
    """Test if expert policy can be loaded and used."""
    print("\nTesting expert policy loading...")
    
    try:
        expert_policy_file = 'rob831/policies/experts/Ant.pkl'
        
        if not os.path.exists(expert_policy_file):
            print(f"❌ Expert policy file not found: {expert_policy_file}")
            return False
            
        expert_policy = LoadedGaussianPolicy(expert_policy_file)
        print("✅ Expert policy loaded successfully")
        
        # Test expert policy with random observation
        env = gym.make('Ant-v2')
        obs = env.reset()
        action = expert_policy.get_action(obs)
        print(f"✅ Expert policy can generate actions, action shape: {action.shape}")
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Expert policy test failed: {e}")
        return False

def test_video_generation():
    """Test video generation capability."""
    print("\nTesting video generation...")
    
    try:
        # Create videos directory under hw1
        videos_dir = os.path.join(os.path.dirname(__file__), 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        
        # 支持的环境和policy文件映射
        env_choices = {
            'Ant-v2': ('rob831/policies/experts/Ant.pkl', 'ant_test.mp4'),
            'Hopper-v2': ('rob831/policies/experts/Hopper.pkl', 'hopper_test.mp4'),
            'Walker2d-v2': ('rob831/policies/experts/Walker2d.pkl', 'walker2d_test.mp4'),
            'HalfCheetah-v2': ('rob831/policies/experts/HalfCheetah.pkl', 'halfcheetah_test.mp4'),
            'Humanoid-v2': ('rob831/policies/experts/Humanoid.pkl', 'humanoid_test.mp4'),
        }

        print("请选择环境:")
        for idx, env_name in enumerate(env_choices.keys()):
            print(f"  {idx+1}. {env_name}")
        choice = input("输入序号 (如1): ").strip()
        try:
            choice_idx = int(choice) - 1
            env_name = list(env_choices.keys())[choice_idx]
        except Exception:
            print("输入无效，默认使用 Ant-v2")
            env_name = 'Ant-v2'

        policy_file, video_filename = env_choices[env_name]
        print(f"选择环境: {env_name}")
        print(f"使用policy文件: {policy_file}")

        env = gym.make(env_name)
        expert_policy = LoadedGaussianPolicy(policy_file)

        # Collect a short trajectory with video
        print("Collecting trajectory with expert policy...")
        paths = utils.sample_n_trajectories(
            env, expert_policy,
            1,  # ntraj
            max_path_length=100,
            render=True
        )

        print(f"✅ Video trajectory collected, path length: {len(paths[0]['observation'])}")
        print(f"✅ Video frames shape: {paths[0]['image_obs'].shape if 'image_obs' in paths[0] else 'No video frames'}")

        # 保存视频为 mp4 文件到 hw1/videos 目录，方便监管和管理
        try:
            import imageio
            video_path = os.path.join(videos_dir, video_filename)
            video_frames = paths[0]['image_obs']
            imageio.mimsave(video_path, video_frames, fps=20)
            print(f"✅ Video saved to {video_path}")
        except Exception as e:
            print(f"❌ Failed to save video: {e}")

        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Video generation test failed: {e}")
        return False

def test_storage_estimation():
    """Estimate storage requirements for videos."""
    print("\nEstimating video storage requirements...")
    
    try:
        env = gym.make('Ant-v2')
        expert_policy_file = 'rob831/policies/experts/Ant.pkl'
        expert_policy = LoadedGaussianPolicy(expert_policy_file)
        
        # Collect a short trajectory to estimate size
        paths = utils.sample_n_trajectories(env, expert_policy, 1, 40, True)
        
        if 'image_obs' in paths[0]:
            video_frames = paths[0]['image_obs']
            frame_size = video_frames.nbytes
            frame_count = len(video_frames)
            
            print(f"Single trajectory video stats:")
            print(f"  - Frame count: {frame_count}")
            print(f"  - Frame shape: {video_frames.shape}")
            print(f"  - Total size: {frame_size / (1024*1024):.2f} MB")
            
            # Estimate for full training
            print(f"\nEstimated storage for full training:")
            print(f"  - BC (1 iter): ~{2 * frame_size / (1024*1024):.2f} MB")
            print(f"  - DAgger (10 iter): ~{20 * frame_size / (1024*1024):.2f} MB")
            print(f"  - All environments: ~{5 * 20 * frame_size / (1024*1024):.2f} MB")
        else:
            print("❌ No video frames found in trajectory")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Storage estimation failed: {e}")
        return False

def test_performance_metrics():
    """Test how to check performance metrics."""
    print("\nTesting performance metrics...")
    
    try:
        # Load expert data to see what performance looks like
        expert_data_file = 'rob831/expert_data/expert_data_Ant-v2.pkl'
        
        if not os.path.exists(expert_data_file):
            print(f"❌ Expert data not found: {expert_data_file}")
            return False
            
        import pickle
        with open(expert_data_file, 'rb') as f:
            expert_paths = pickle.load(f)
        
        # Calculate expert performance
        expert_returns = [path["reward"].sum() for path in expert_paths]
        expert_lengths = [len(path["reward"]) for path in expert_paths]
        
        print(f"Expert performance stats:")
        print(f"  - Number of trajectories: {len(expert_paths)}")
        print(f"  - Average return: {np.mean(expert_returns):.2f} ± {np.std(expert_returns):.2f}")
        print(f"  - Average episode length: {np.mean(expert_lengths):.2f}")
        print(f"  - Target BC performance (30%): {0.3 * np.mean(expert_returns):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        return False

def main():
    print("🧪 HW1 Environment and Video Test Suite")
    print("="*50)
    
    # Change to hw1 directory
    os.chdir('/home/shuai/Projects/16831-S25/hw1')
    
    tests = [
        test_environment_rendering,
        test_expert_policy,
        test_video_generation,
        test_storage_estimation,
        test_performance_metrics,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "="*50)
    print("📊 Test Summary:")
    test_names = [
        "Environment Rendering",
        "Expert Policy Loading", 
        "Video Generation",
        "Storage Estimation",
        "Performance Metrics"
    ]
    
    for name, result in zip(test_names, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'🎉 All tests passed!' if all_passed else '⚠️ Some tests failed'}")
    
    if all_passed:
        print("\n🚀 Ready to start training!")
        print("Next steps:")
        print("1. Run BC training: python rob831/scripts/run_hw1.py ...")
        print("2. Monitor with tensorboard: tensorboard --logdir data/")
        print("3. Check videos in tensorboard for visual debugging")

if __name__ == "__main__":
    main()
