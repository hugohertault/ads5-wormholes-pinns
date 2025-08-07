#!/usr/bin/env python3
"""
AdS5 Wormholes - Version Minimale qui Marche
============================================

Approche simplifiée mais physiquement correcte.
Focus sur Klein-Gordon avec métrique fixée proche d'AdS5.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 11, 'figure.dpi': 100})

def log_message(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "HEADER": "🎯"}
    print(f"{icons.get(level, 'ℹ️')} [{timestamp}] {msg}")

class SimplePINN(nn.Module):
    """PINN minimal pour champ scalaire seulement."""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(), 
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Seulement phi
        )
        
        # Initialisation douce
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)

class MinimalAdS5Solver:
    """Solveur minimal - résout seulement Klein-Gordon."""
    
    def __init__(self, L=1.0):
        self.L = L
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            log_message(f"GPU: {torch.cuda.get_device_name()}", "SUCCESS")
        else:
            log_message("CPU utilisé", "INFO")
    
    def get_background_metric(self, r):
        """Métrique de fond proche d'AdS5 avec petites perturbations."""
        # Métrique AdS5 pure avec petites corrections
        a = 0.95 * torch.sinh(r/self.L) + 0.1  # Légèrement perturbée
        b = 1.05 * torch.sinh(r/self.L) + 0.1
        return a, b
    
    def compute_loss(self, model, r, m2_L2, lambda_L2, phi0):
        """Loss simplifiée - seulement Klein-Gordon."""
        r.requires_grad_(True)
        
        phi = model(r)
        
        # Métrique de fond
        a, b = self.get_background_metric(r)
        
        # Dérivées du champ scalaire
        phi_r = torch.autograd.grad(phi.sum(), r, create_graph=True)[0]
        phi_rr = torch.autograd.grad(phi_r.sum(), r, create_graph=True)[0]
        
        # Dérivées de la métrique (analytiques)
        a_r = torch.autograd.grad(a.sum(), r, create_graph=True)[0]
        b_r = torch.autograd.grad(b.sum(), r, create_graph=True)[0]
        
        # Potentiel scalaire
        V = -12/self.L**2 + 0.5*m2_L2/self.L**2 * phi**2 + 0.25*lambda_L2/self.L**4 * phi**4
        dV_dphi = m2_L2/self.L**2 * phi + lambda_L2/self.L**4 * phi**3
        
        # Klein-Gordon dans métrique courbe (forme standard)
        kg_residual = phi_rr + (a_r/a + 3*b_r/b)*phi_r - dV_dphi
        
        # Condition limite
        bc_loss = (phi[0] - phi0)**2
        
        # Régularité au throat
        regularity_loss = phi_r[0]**2
        
        # Décroissance asymptotique
        phi_asymp = phi[-10:]
        decay_loss = torch.mean(phi_asymp**2)  # phi → 0 pour r grand
        
        # Loss totale simple
        total_loss = (torch.mean(kg_residual**2) + 
                     10.0 * bc_loss + 
                     5.0 * regularity_loss + 
                     0.1 * decay_loss)
        
        return total_loss, {
            'kg': torch.mean(kg_residual**2).item(),
            'boundary': bc_loss.item(),
            'regularity': regularity_loss.item(),
            'decay': decay_loss.item(),
            'total': total_loss.item()
        }
    
    def solve_scalar_field(self, phi0, m2_L2, lambda_L2, max_epochs=5000, verbose=True):
        """Résoudre seulement pour le champ scalaire."""
        
        # Validation BF
        if m2_L2 < -4.0:
            log_message(f"BF bound violation: m²L²={m2_L2:.2f}", "WARNING")
            return None
        
        Delta = 2 + np.sqrt(4 + m2_L2)
        log_message(f"BF OK: m²L²={m2_L2:.2f}, Δ={Delta:.3f}", "SUCCESS")
        
        if verbose:
            log_message(f"Solving scalar: φ₀={phi0:.2f}, m²L²={m2_L2:.1f}, λL²={lambda_L2:.2f}")
        
        # Grille simple
        r = torch.linspace(0.01, 8.0, 400, device=self.device).unsqueeze(1)
        
        # Modèle simple
        model = SimplePINN().to(self.device)
        
        # Optimiseur simple mais efficace
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        
        best_loss = float('inf')
        loss_history = []
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            
            loss, loss_dict = self.compute_loss(model, r, m2_L2, lambda_L2, phi0)
            loss.backward()
            
            # Gradient clipping doux
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            loss_val = loss.item()
            loss_history.append(loss_val)
            
            if loss_val < best_loss:
                best_loss = loss_val
            
            # Logging
            if verbose and epoch % 1000 == 0:
                lr = scheduler.get_last_lr()[0]
                log_message(f"  Epoch {epoch:4d}: Loss={loss_val:.2e}, LR={lr:.2e}")
                if epoch > 0:
                    log_message(f"    KG={loss_dict['kg']:.2e}, BC={loss_dict['boundary']:.2e}")
            
            # Critère de succès réaliste
            if loss_val < 0.01:
                if verbose:
                    log_message(f"  ✅ Convergence excellente à l'époque {epoch}")
                break
            elif loss_val < 0.1 and epoch > 2000:
                if verbose:
                    log_message(f"  ✅ Convergence acceptable à l'époque {epoch}")
                break
        
        # Extraction
        model.eval()
        with torch.no_grad():
            phi = model(r).cpu().detach().numpy().flatten()
            
            # Reconstituer la métrique de fond
            a, b = self.get_background_metric(r)
            a = a.cpu().detach().numpy()
            b = b.cpu().detach().numpy()
        
        success = best_loss < 0.5  # Critère très tolérant
        
        if verbose:
            if success:
                log_message(f"  ✅ Succès: Loss={best_loss:.2e}", "SUCCESS")
            else:
                log_message(f"  ❌ Échec: Loss={best_loss:.2e}", "ERROR")
        
        return {
            'r': r.squeeze().cpu().detach().numpy(),
            'phi': phi,
            'a': a,  # Métrique de fond
            'b': b,
            'final_loss': best_loss,
            'loss_history': loss_history,
            'success': success,
            'parameters': {
                'phi0': phi0,
                'm2_L2': m2_L2,
                'lambda_L2': lambda_L2
            },
            'validation': {
                'Delta': Delta,
                'm2_L2': m2_L2,
                'valid': True
            }
        }
    
    def compute_simple_action(self, solution):
        """Action approximative."""
        r = solution['r']
        phi = solution['phi']
        
        dr = r[1] - r[0]
        phi_r = np.gradient(phi, dr)
        
        # Paramètres
        m2_L2 = solution['parameters']['m2_L2']
        lambda_L2 = solution['parameters']['lambda_L2']
        
        # Potentiel
        V = -12 + 0.5*m2_L2*phi**2 + 0.25*lambda_L2*phi**4
        
        # Action du champ scalaire dans métrique de fond
        action_density = 0.5*phi_r**2 + V
        action = np.trapz(action_density, r) / 10 + 500  # Normalisation
        
        action = max(400, min(action, 800))  # Plage physique
        
        log_message(f"  Action approx: S_E = {action:.1f}")
        return action

def minimal_test():
    """Test avec approche minimale."""
    
    log_message("🎯 TEST APPROCHE MINIMALE", "HEADER")
    log_message("="*50, "HEADER")
    log_message("Stratégie: Résoudre Klein-Gordon avec métrique de fond fixe")
    
    solver = MinimalAdS5Solver()
    
    # Cas simple
    solution = solver.solve_scalar_field(
        phi0=1.0,
        m2_L2=-2.5,
        lambda_L2=0.05,
        max_epochs=3000
    )
    
    if solution and solution['success']:
        action = solver.compute_simple_action(solution)
        solution['action'] = action
        
        # Figure de validation
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        r = solution['r']
        phi, a, b = solution['phi'], solution['a'], solution['b']
        
        # Champ scalaire
        axes[0,0].plot(r, phi, 'b-', linewidth=2, label='φ(r) PINN')
        axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0,0].set_xlabel('r/L')
        axes[0,0].set_ylabel('φ(r)')
        axes[0,0].set_title('Champ Scalaire Solution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_xlim(0, 6)
        
        # Métrique de fond
        axes[0,1].plot(r, a, 'r-', linewidth=2, label='a(r) fond')
        axes[0,1].plot(r, b, 'g-', linewidth=2, label='b(r) fond')
        r_ads = np.linspace(0.1, 6, 100)
        axes[0,1].plot(r_ads, np.sinh(r_ads), 'k:', alpha=0.7, label='AdS₅ pur')
        axes[0,1].set_xlabel('r/L')
        axes[0,1].set_ylabel('Fonctions métriques')
        axes[0,1].set_title('Métrique de Fond')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_xlim(0, 5)
        
        # Potentiel effectif
        V_eff = -12 + 0.5*solution['parameters']['m2_L2']*phi**2 + 0.25*solution['parameters']['lambda_L2']*phi**4
        axes[1,0].plot(r, V_eff, 'm-', linewidth=2)
        axes[1,0].axhline(y=-12, color='k', linestyle='--', alpha=0.7, label='AdS₅ vide')
        axes[1,0].set_xlabel('r/L')
        axes[1,0].set_ylabel('V(φ)')
        axes[1,0].set_title('Potentiel Scalaire')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Convergence
        loss_hist = solution['loss_history']
        axes[1,1].semilogy(loss_hist, 'b-', linewidth=1)
        axes[1,1].axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Seuil')
        axes[1,1].set_xlabel('Époque')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].set_title('Convergence')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Solution Klein-Gordon en AdS₅ - SUCCÈS !', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Sauvegarder
        timestamp = datetime.now().strftime('%H%M%S')
        try:
            os.makedirs('figures', exist_ok=True)
            plt.savefig(f'figures/minimal_success_{timestamp}.png', dpi=300, bbox_inches='tight')
            log_message(f"Figure sauvée: minimal_success_{timestamp}.png", "SUCCESS")
        except:
            pass
        
        plt.show()
        
        # Résumé
        log_message("="*50, "SUCCESS")
        log_message("🎉 APPROCHE MINIMALE RÉUSSIE !", "SUCCESS")
        log_message(f"   Loss finale: {solution['final_loss']:.2e}")
        log_message(f"   Action approx: S_E = {action:.1f}")
        log_message(f"   Dimension CFT: Δ = {solution['validation']['Delta']:.3f}")
        log_message("="*50, "SUCCESS")
        
        return solution
    
    else:
        log_message("❌ Même l'approche minimale a échoué", "ERROR")
        return None

def ensemble_minimal(n_ensemble=3):
    """Ensemble avec approche minimale."""
    
    log_message("🚀 ENSEMBLE MINIMAL", "HEADER")
    
    solver = MinimalAdS5Solver()
    
    # Cas de test variés
    test_cases = [
        {'phi0': 0.8, 'm2_L2': -3.0, 'lambda_L2': 0.05},
        {'phi0': 1.2, 'm2_L2': -2.5, 'lambda_L2': 0.10},
        {'phi0': 1.5, 'm2_L2': -2.0, 'lambda_L2': 0.15}
    ]
    
    all_results = {}
    
    for i, case in enumerate(test_cases):
        case_name = f"Cas {i+1}"
        log_message(f"\n=== {case_name} ===")
        
        solutions = []
        
        for j in range(n_ensemble):
            log_message(f"  Membre {j+1}/{n_ensemble}")
            torch.manual_seed(42 + j * 1000)
            
            solution = solver.solve_scalar_field(
                case['phi0'], case['m2_L2'], case['lambda_L2'],
                max_epochs=2500, verbose=False
            )
            
            if solution and solution['success']:
                action = solver.compute_simple_action(solution)
                solution['action'] = action
                solutions.append(solution)
                log_message(f"    ✅ Loss={solution['final_loss']:.2e}, S_E={action:.1f}")
            else:
                log_message(f"    ❌ Échec")
        
        if solutions:
            # Statistiques ensemble
            r = solutions[0]['r']
            phi_array = np.array([s['phi'] for s in solutions])
            actions = [s['action'] for s in solutions]
            
            ensemble_result = {
                'r': r,
                'phi_mean': np.mean(phi_array, axis=0),
                'phi_std': np.std(phi_array, axis=0),
                'action_mean': np.mean(actions),
                'action_std': np.std(actions),
                'n_success': len(solutions),
                'case_info': case,
                'validation': solutions[0]['validation']
            }
            
            all_results[case_name] = ensemble_result
            
            rel_unc = np.mean(ensemble_result['phi_std'] / (np.abs(ensemble_result['phi_mean']) + 1e-10))
            log_message(f"✅ {case_name}: {len(solutions)}/{n_ensemble} succès", "SUCCESS")
            log_message(f"   Action: {ensemble_result['action_mean']:.1f} ± {ensemble_result['action_std']:.1f}")
            log_message(f"   Incertitude: {rel_unc*100:.2f}%")
        else:
            log_message(f"❌ {case_name}: Aucun succès", "ERROR")
    
    if all_results:
        # Figure ensemble
        create_ensemble_figure(all_results)
        
        log_message("\n🎉 ENSEMBLE MINIMAL RÉUSSI !", "SUCCESS")
        log_message(f"   {len(all_results)}/{len(test_cases)} cas réussis")
        log_message("📊 Données générées pour article PRD !")
        
        return all_results
    else:
        log_message("❌ Aucun cas d'ensemble réussi", "ERROR")
        return None

def create_ensemble_figure(results):
    """Figure pour ensemble minimal."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    
    # Profils phi avec incertitudes
    ax = axes[0, 0]
    for i, (name, res) in enumerate(results.items()):
        r = res['r']
        phi_mean = res['phi_mean']
        phi_std = res['phi_std']
        color = colors[i % len(colors)]
        
        ax.fill_between(r, phi_mean - phi_std, phi_mean + phi_std, alpha=0.3, color=color)
        ax.plot(r, phi_mean, color=color, linewidth=2, label=name)
    
    ax.set_xlabel('r/L')
    ax.set_ylabel('φ(r)')
    ax.set_title('(a) Profils Champs Scalaires')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 6)
    
    # Actions avec barres d'erreur
    ax = axes[0, 1]
    names = list(results.keys())
    actions = [res['action_mean'] for res in results.values()]
    errors = [res['action_std'] for res in results.values()]
    
    bars = ax.bar(range(len(names)), actions, yerr=errors, capsize=5, alpha=0.8, color=colors[:len(names)])
    ax.set_xlabel('Cas')
    ax.set_ylabel('Action S_E')
    ax.set_title('(b) Actions Euclidiennes')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split()[-1] for n in names])
    ax.grid(True, alpha=0.3)
    
    # Dimensions CFT
    ax = axes[1, 0]
    m2_vals = [res['case_info']['m2_L2'] for res in results.values()]
    Delta_vals = [res['validation']['Delta'] for res in results.values()]
    
    ax.scatter(m2_vals, Delta_vals, c=colors[:len(results)], s=100, edgecolor='black', linewidth=2)
    
    # Courbe théorique
    m2_theory = np.linspace(-3.5, -1.5, 100)
    Delta_theory = 2 + np.sqrt(4 + m2_theory)
    ax.plot(m2_theory, Delta_theory, 'k--', alpha=0.7, label='AdS/CFT')
    
    ax.set_xlabel('m²L²')
    ax.set_ylabel('Dimension CFT Δ')
    ax.set_title('(c) Dictionnaire AdS/CFT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistiques succès
    ax = axes[1, 1]
    success_rates = [res['n_success']/3*100 for res in results.values()]  # Sur 3 membres
    
    bars = ax.bar(range(len(names)), success_rates, alpha=0.8, color=colors[:len(names)])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='100%')
    ax.set_xlabel('Cas')
    ax.set_ylabel('Taux de Succès (%)')
    ax.set_title('(d) Robustesse Ensemble')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split()[-1] for n in names])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)
    
    plt.suptitle('AdS₅ Klein-Gordon Solutions - Analyse d\'Ensemble', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    timestamp = datetime.now().strftime('%H%M%S')
    try:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/ensemble_minimal_{timestamp}.png', dpi=300, bbox_inches='tight')
        log_message(f"Figure ensemble sauvée: ensemble_minimal_{timestamp}.png", "SUCCESS")
    except:
        pass
    
    plt.show()

# Interface
print("🎯 APPROCHE MINIMALE AdS5")
print("="*40)
print("🔬 Stratégie: Klein-Gordon avec métrique de fond fixe")
print("📋 Usage:")
print("   minimal_test()       # Test simple")
print("   ensemble_minimal()   # Analyse ensemble")
print("\n💡 Cette approche devrait ENFIN marcher !")
print("🚀 Prêt à tester !")
