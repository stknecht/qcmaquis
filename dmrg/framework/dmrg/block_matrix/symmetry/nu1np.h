/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2013 Institute for Theoretical Physics, ETH Zurich
 *               2013-2013 by Bela Bauer <bauerb@phys.ethz.ch>
 *                            Sebastian Keller <sebkelleb@phys.ethz.ch>
 *
 * This software is part of the ALPS Applications, published under the ALPS
 * Application License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 *
 * You should have received a copy of the ALPS Application License along with
 * the ALPS Applications; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef SYMMETRY_NU1NP_H
#define SYMMETRY_NU1NP_H

#include "utils/io.hpp"
#include <vector>
#include <list>

#include <boost/lexical_cast.hpp>
#include <boost/functional/hash.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>

#include <alps/numeric/matrix.hpp>

#include <dmrg/utils/BaseParameters.h>

#include <dmrg/block_matrix/symmetry/nu1pg.h>
//#include <dmrg/block_matrix/symmetry/dg_tables.h>

template<int N, class S>
class NU1NP;

template<int N, class S>
class NU1ChargeNP : public NU1ChargePG<N, S>
{
    typedef NU1ChargePG<N, S> base;

public:
    NU1ChargeNP(S init = 0) : base(init) {}
    NU1ChargeNP(boost::array<S, N> const & rhs) : base(rhs) {}

    S * begin() { return base::begin(); }
    S * end() { return base::end(); }

    S const * begin() const { return base::begin(); }
    S const * end() const { return base::end(); }

    S & operator[](std::size_t p) { return base::operator[](p); }
    S const & operator[](std::size_t p) const { return base::operator[](p); }

    template<class Archive>
    void save(Archive & ar) const
    {
        base::save(ar);
    }

    template<class Archive>
    void load(Archive & ar)
    {
        base::load(ar);
    }

    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        base::serialize(ar, version);
    }

};

namespace boost {
    template <int N, class S>
    class hash<NU1ChargeNP<N, S> >{
        public :
            size_t operator()(NU1ChargeNP<N, S> const &Charge ) const {
                return hash<NU1ChargePG<N, S> >()(Charge);
            }
    };

    template <int N, class S>
    class hash<std::pair<NU1ChargeNP<N, S>, NU1ChargeNP<N, S> > >{
        public :
            size_t operator()(std::pair<NU1ChargeNP<N, S>, NU1ChargeNP<N, S> > const &Pair_of_charge ) const {
                return hash<std::pair<NU1ChargePG<N, S>, NU1ChargePG<N, S> > >()(Pair_of_charge);
            }
    };

}

template<int N, class S>
inline bool operator<(NU1ChargeNP<N, S> const & a, NU1ChargeNP<N, S> const & b)
{
    return tpl_ops_pg_<N, 0>().operator_lt(a.begin(), b.begin());
}

template<int N, class S>
inline bool operator>(NU1ChargeNP<N, S> const & a, NU1ChargeNP<N, S> const & b)
{
    return tpl_ops_pg_<N, 0>().operator_gt(a.begin(), b.begin());
}

template<int N, class S>
inline bool operator==(NU1ChargeNP<N, S> const & a, NU1ChargeNP<N, S> const & b)
{
    return tpl_ops_pg_<N, 0>().operator_eq(a.begin(), b.begin());
}

template<int N, class S>
inline bool operator!=(NU1ChargeNP<N, S> const & a, NU1ChargeNP<N, S> const & b)
{
    return !(a==b);
}

template<int N, class S>
NU1ChargeNP<N, S> operator+(NU1ChargeNP<N, S> const & a,
                       NU1ChargeNP<N, S> const & b)
{
    NU1ChargeNP<N, S> ret;
    tpl_arith_<NU1NP<N, S>, N, 0>().operator_plus(a.begin(), b.begin(), ret.begin());
    return ret;
}


template<int N, class S>
NU1ChargeNP<N, S> operator-(NU1ChargeNP<N, S> const & rhs)
{
    NU1ChargeNP<N, S> ret;
    tpl_arith_<NU1NP<N,S>, N, 0>().operator_uminus(rhs.begin(), ret.begin());
    return ret;
}

template<int N, class S>
NU1ChargeNP<N, S> operator/(NU1ChargeNP<N, S> const & a, int n)
{
    NU1ChargeDG<N, S> ret;
    tpl_arith_<NU1NP<N,S>, N, 0>().operator_div(a.begin(), ret.begin(), n);
    return ret;
}

template<int N, class S = int>
class NU1NP
{
    template<class G, int A, int B> friend struct tpl_arith_;
public:
    typedef S subcharge;
    typedef NU1ChargeNP<N, S> charge;
    typedef std::vector<charge> charge_v;
private:
    static alps::numeric::matrix<S> mult_table;
    static std::vector<S> adjoin_table;
    static std::size_t group_id;
    static subcharge max_irrep;
public:
    static const charge IdentityCharge;
    static const bool finite = false;

    static subcharge adjoin(subcharge I)
    {
        return adjoin_table[I];
    }

    static subcharge particleNumber(charge rhs) { return std::accumulate(rhs.begin(), &rhs[N], 0); }

    static subcharge & irrep(charge & rhs) { return rhs[N]; }
    static subcharge const & irrep(charge const & rhs) { return rhs[N]; }

    static charge fuse(charge a, charge b)
    {
        return a+b;
    }

    template<int R> static charge fuse(boost::array<charge, R> const & v)
    {
        charge ret = v[0];
        for (int i = 1; i < R; ++i)
            ret = fuse(ret, v[i]);
        return ret;
    }


    static subcharge const get_max_irrep()
    {
        return max_irrep;
    }

};

template<int N, class S> const typename NU1NP<N,S>::charge NU1NP<N,S>::IdentityCharge = typename NU1NP<N,S>::charge();
//template<int N, class S> alps::numeric::matrix<S> NU1NP<N,S>::mult_table = default_dg_mult_table<S>();
//template<int N, class S> std::vector<S> NU1NP<N,S>::adjoin_table = default_dg_adjoin_table<S>();
template<int N, class S> std::size_t NU1NP<N,S>::group_id = 0;
template<int N, class S> typename NU1NP<N,S>::subcharge NU1NP<N,S>::max_irrep = 0;

typedef NU1NP<1> U1NP;

#endif
