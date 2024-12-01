import rlgym
from rlgym import make
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.obs_builders import DefaultObs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import optuna

from tensorboardX import SummaryWriter  # Use tensorboardX instead

import numpy as np
import os
import pickle

# 1. Définition des Conditions de Terminaison Personnalisées
from rlgym.utils import BaseTerminationCondition

class CustomTimeoutCondition(BaseTerminationCondition):
    def __init__(self, timeout=300):
        self.timeout = timeout

    def reset(self, initial_state):
        self.current_step = 0

    def is_terminal(self, state):
        self.current_step += 1
        return self.current_step >= self.timeout

class CustomGoalScoredCondition(BaseTerminationCondition):
    def is_terminal(self, state):
        # Exemple simple : terminer si un but est marqué
        # Vous devrez adapter cette condition en fonction des attributs de 'state'
        # Par exemple, si 'state' contient les informations sur les buts, ajustez en conséquence
        # L'exemple ci-dessous est fictif et doit être adapté
        # Remplacez 'state.ball.position[0] > 1000' par une condition réelle basée sur l'état
        return state.ball.position[0] > 1000  # Exemple fictif

# 2. Configuration de l'Environnement rlgym
def create_rlgym_env():
    env = make(
        reward_fn=DefaultReward(),
        terminal_conditions=[CustomTimeoutCondition(timeout=300), CustomGoalScoredCondition()],
        state_setter=DefaultState(),
        obs_builder=DefaultObs()
    )
    return env

# 3. Définition des Modèles Individuels
# 3.1 CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 512)  # Ajustez selon la taille de l'entrée

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x

# 3.2 Mécanisme d'Attention (Transformers)
class AttentionModule(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x: [batch_size, seq_length, feature_dim]
        attn_output, _ = self.attention(x, x, x)
        x = F.relu(self.fc(attn_output))
        return x

# 3.3 Modèles Secondaires (Random Forests, LightGBM, SVM)
class SecondaryModels:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        self.svm = SVC(probability=True, random_state=42)
    
    def train(self, X, y):
        self.rf.fit(X, y)
        self.lgbm.fit(X, y)
        self.svm.fit(X, y)
    
    def predict_proba(self, X):
        rf_preds = self.rf.predict_proba(X)
        lgbm_preds = self.lgbm.predict_proba(X)
        svm_preds = self.svm.predict_proba(X)
        # Combiner les prédictions par moyenne
        combined_preds = (rf_preds + lgbm_preds + svm_preds) / 3
        return combined_preds

# 4. Agent Hybride avec Ensemble Learning
class HybridAgent(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(HybridAgent, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels)
        self.attention = AttentionModule(feature_dim=512)
        self.fc = nn.Linear(512, num_actions)
        self.secondary_models = SecondaryModels()
    
    def forward(self, x):
        features = self.cnn(x)
        # Ajouter une dimension de séquence pour le Transformer
        features = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        attention_out = self.attention(features).squeeze(1)  # [batch_size, feature_dim]
        action_logits = self.fc(attention_out)
        return action_logits
    
    def train_secondary_models(self, X, y):
        self.secondary_models.train(X, y)
    
    def predict_secondary_models(self, X):
        return self.secondary_models.predict_proba(X)

# 5. Knowledge Distillation
class StudentModel(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(StudentModel, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels)
        self.fc = nn.Linear(512, num_actions)
    
    def forward(self, x):
        features = self.cnn(x)
        action_logits = self.fc(features)
        return action_logits

def knowledge_distillation(student, teacher, dataloader, optimizer, criterion, device):
    student.train()
    teacher.eval()
    for batch in dataloader:
        inputs, _ = batch
        inputs = inputs.to(device)
        with torch.no_grad():
            teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)
        loss = criterion(student_outputs, teacher_outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 6. Hyperparameter Tuning avec Optuna
def evaluate_agent(agent, env, num_episodes=10):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        all_rewards.append(total_reward)
    return np.mean(all_rewards)

def optimize_agent(trial):
    # Hyperparamètres à optimiser
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_int('n_steps', 128, 2048)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gamma = trial.suggest_uniform('gamma', 0.8, 0.9999)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
    
    # Créer l'environnement vectorisé
    vec_env = make_vec_env(lambda: create_rlgym_env(), n_envs=4)
    
    # Définir les paramètres du modèle PPO
    policy_kwargs = dict(
        features_extractor_class=CustomHybridExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        activation_fn=nn.ReLU,
    )
    
    # Instancier l'agent PPO avec notre modèle hybride
    agent = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        tensorboard_log="./ppo_rlgym_tensorboard/"
    )
    
    # Entraîner l'agent
    agent.learn(total_timesteps=50000)  # Ajustez selon vos ressources
    
    # Évaluer l'agent
    mean_reward = evaluate_agent(agent, vec_env)
    
    # Nettoyage
    vec_env.close()
    
    return mean_reward

# 7. Surveillance des Performances avec TensorBoard
def setup_tensorboard(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def log_metrics(writer, step, metrics):
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)

# 8. Définir l'Extracteur de Caractéristiques Personnalisé
class CustomHybridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomHybridExtractor, self).__init__(observation_space, features_dim)
        self.cnn = CNNFeatureExtractor(input_channels=observation_space.shape[0])
        self.attention = AttentionModule(feature_dim=512)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, features_dim)
    
    def forward(self, observations):
        x = self.cnn(observations)
        x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        x = self.attention(x)
        x = x.squeeze(1)  # [batch_size, feature_dim]
        x = self.fc(x)
        return x

# 9. Fonction Principale
def main():
    # Configuration de TensorBoard
    log_dir = "./ppo_rlgym_tensorboard/"
    writer = setup_tensorboard(log_dir)
    
    # Optimisation des hyperparamètres avec Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=20)
    
    print("Meilleurs hyperparamètres:", study.best_params)
    
    # Entraînement final avec les meilleurs hyperparamètres
    best_params = study.best_params
    vec_env = make_vec_env(lambda: create_rlgym_env(), n_envs=4)
    
    policy_kwargs = dict(
        features_extractor_class=CustomHybridExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        activation_fn=nn.ReLU,
    )
    
    agent = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=best_params['learning_rate'],
        n_steps=best_params['n_steps'],
        batch_size=best_params['batch_size'],
        gamma=best_params['gamma'],
        ent_coef=best_params['ent_coef'],
        tensorboard_log=log_dir
    )
    
    # Entraîner l'agent
    agent.learn(total_timesteps=200000, callback=None)  # Ajustez selon vos ressources
    
    # Sauvegarder le modèle
    agent.save("ppo_rlgym_hybrid_agent")
    
    # Évaluer l'agent
    mean_reward = evaluate_agent(agent, vec_env)
    print(f"Récompense moyenne après entraînement: {mean_reward}")
    
    # Fermer l'environnement
    vec_env.close()
    
    # Fermer TensorBoard
    writer.close()

if __name__ == "__main__":
    main()